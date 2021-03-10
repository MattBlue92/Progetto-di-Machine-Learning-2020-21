from transformers import RobertaModel
from MyLongformerSelfAttention import MyLongformerSelfAttention
class MyLongformer(RobertaModel): #il longformer eredità da roberta ed è un roberta
    def __init__(self, config):
        """
        Args:
        config: oggetto che contiene delle info di configurazione come quale metodo di calcolo
        usare per l'attenzione se quella classica indicata con n2 oppure tvm. Io implemento solo
        la versione tvm con il cuda kernel
        :param config:
        """
        super(MyLongformer, self).__init__(config)
        for i, enc_layer in enumerate(self.encoder.layer):
            enc_layer.attention.self = MyLongformerSelfAttention(config, layer_id = i)


