import torch
from torch import nn
import math
from myLongFormerCode.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations

class MyLongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        """

        :param config: Oggetto di configurazione simile a quello di roberta
         contenente alcune informazioni come il numero di teste
        di attenzione h (num_attention_heads) e la dimensione di una rappresentazione r (hidden_size
        dim (n*hd_v)->hd_v = 768)
        :param layer_id: intero che identifica la posizione del layer di tipo encoder nello stack di encoder
        """
        super(MyLongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads) #dimensione di una matrice di attenzione
        self.embed_dim = config.hidden_size # dimensione di una rappresentazione fissata a 768 con bert-base

        #matrici delle query, key e value locali per il calcolo della matrice di attenzione con lo sliding window
        # le matrici sono dei layer linear di pytorch che implementano y = xA^T + by o y = xA^T.
        self.query_s = nn.Linear(config.hidden_size, self.embed_dim) # aspetta nella teoria Q,K,V sono di dim n*d ma qui  sembra siano di n*h(n*d) o in multihead n*d_k
        self.key_s = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_s = nn.Linear(config.hidden_size, self.embed_dim)

        # matrici delle query, key e value globali per la global attention
        # le matrici sono dei layer linear di pytorch che implementano y = xA^T + by o y = xA^T.
        self.query_g = nn.Linear(config.hidden_size, self.embed_dim)  # aspetta nella teoria Q,K,V sono di dim n*d ma qui  sembra siano di n*h(n*d) o in multihead n*d_k
        self.key_g = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_g = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id

        self.attention_window = config.attention_window[self.layer_id] # ritorna l'attention window per il layer_id corrente
        self.attention_dilation = config.attention_dilation[self.layer_id] # idem con patate
        self.attention_mode = config.attention_mode #modalità di attenzione ?? n2, tvm, chunk??
        self.autoregressive = config.autoregressive

        assert self.attention_window > 0 #controlla che la dim della finestra w sia maggiore di 0
        assert self.attention_dilation > 0 #controlla che il valore di dilatazione d sia maggiore di 0
        assert self.attention_mode == 'tvm'

        """
        # non necessario se uso tvm
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported
        """

    def forward(self, hidden_states,
                attention_mask = None,
                head_mask = None,
                encoder_hidden_states = None,
                encoder_attention_mask = None,
                output_attention = False):

        """

        :param hidden_states: output di layer di i-1 encoder che diventa l'input dell'encoder i-encoder dim  [batch_sample,num tokens, 768]
        :param attention_mask: maschera di attenzione per capire quali sono i token diversi dal token [PAD]
        :param head_mask:  ???
        :param encoder_hidden_states: ???
        :param encoder_attention_mask: ???
        :param output_attention: ???
        :return:
        """

        assert encoder_hidden_states is None, "`encoder_hidden_states` non è supportata e dovrebbe essere None"
        assert encoder_attention_mask is None, "`encoder_attention_mask`  non è supportata e dovrebbe essere None"

        if attention_mask is not None:
            print(attention_mask.size())
            print(attention_mask.numel())
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1) #togli la dimensione 2 e 1 quando  sono unitarie
            print(attention_mask.size())

            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask >0 #  genera una maschera di booleani dove true vuol dire che è un token false invece se [PAD] in quanto 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1) #somma per colonna i true
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max() #ritorna il valore massimo da num_extra_indices_per_batch
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # Per supportare il caso di un numero variabile di attenzione globale nelle righe di un batch,
                # usiamo le seguenti tre maschere di selezione per selezionare gli embedding dell'attenzione globale
                # in un tensore 3d e aggiungilo a `max_num_extra_indices_per_batch`

                # 1) selezionando gli incorporamenti che corrispondono all'attenzione globale
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True) # ritorna un tensore multi-dim
                #che indica gli indici degli elementi diversi zero, per ogni indice crea 3:
                # per batch, per riga, per colonna
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device) #genera un tensore array con interi da 0 a max_num_extra_indices_per_batch
                # maschera che indica quali valori verranno effettivamente paddati
                # la maschera è un tensore multi-dimensione (bacht, tokens, frasi), dove true indica in valore diversa dal token [PAD]
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1) # aggiunge una terza dimensione in fondo al tensore

                # 2) posizione dei valori non di riempimento nell'attenzione globale selezionata
                # ritorna un'insieme di tensori  (uno per batch, uno per token e uno per frase) degli indici diversi da zero
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)

                # 3) posizione dei valori di riempimento nell'attenzione globale selezionata
                # fa il contrario di selection_padding_mask_nonzeros
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)


        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1) # può essere un tensore multi-dimensionale [batch, frase, token], fa una trasposizione
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query_s(hidden_states) # calcola la matrice query locale
        k = self.key_s(hidden_states) # calcola la matrice delle chiavi locale
        v = self.value_s(hidden_states) # calcola la matrice dei valori
        q /= math.sqrt(self.head_dim) # normalizza

        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False) # chiama il metodo forward
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask,
                                                                                    -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0,
                                           False)

            attn_weights += d_mask

        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]
