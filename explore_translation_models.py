# %%
import os
os.environ['XDG_CACHE'] = '/workspace/.cache'
os.environ['HF_HOME']='/workspace/.cache/huggingface'


# %%
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pyonmttok
import ctranslate2
from metrics import *
from huggingface_hub import snapshot_download

# %%
print("Loading translator Models...")
ca_en_model_folder = snapshot_download(repo_id="projecte-aina/mt-aina-ca-en", revision="main")
tokenizer_ca_en = pyonmttok.Tokenizer(
    mode="none", sp_model_path=ca_en_model_folder + "/spm.model"
)
ca_en_model = ctranslate2.Translator(ca_en_model_folder, device="cuda")
en_ca_model_folder = snapshot_download(repo_id="projecte-aina/mt-aina-en-ca", revision="main")
tokenizer_en_ca = pyonmttok.Tokenizer(
    mode="none", sp_model_path=en_ca_model_folder + "/spm.model"
)
en_ca_model = ctranslate2.Translator(en_ca_model_folder, device="cuda")
# %%
def translate_to_english(txt):
    lines = [l for l in txt.split("\n") if l.strip() != ""]

    toks, _ = tokenizer_ca_en.tokenize_batch(lines)
    translated = ca_en_model.translate_batch(toks, max_input_length=2048, max_decoding_length=2048)
    ts = []
    for t in translated:
        t_str = tokenizer_ca_en.detokenize(t.hypotheses[0])
        # That is a bug on the translation outputing twice the translation.
        if len(txt.split(" ")) == 1 and len(t_str.split(" ")) == 2:
            t_str = t_str.split(" ")[0]
        ts.append(t_str)

    return "\n".join(ts)

def translate_to_catalan(txt):
    lines = [l for l in txt.split("\n") if l.strip() != ""]

    toks, _ = tokenizer_en_ca.tokenize_batch(lines)
    translated = en_ca_model.translate_batch(toks)
    ts = []
    for t in translated:
        t_str = tokenizer_en_ca.detokenize(t.hypotheses[0])
        # That is a bug on the translation outputing twice the translation.
        if len(txt.split(" ")) == 1 and len(t_str.split(" ")) == 2:
            t_str = t_str.split(" ")[0]
        ts.append(t_str)

    return "\n".join(ts)
# %%
example = """Algunes teories desenvolupades durant la dècada dels anys 70 van definir possibles vies a través de les quals la desigualtat pot tenir un efecte positiu en el desenvolupament econòmic. Segons una anàlisi del 1955, es pensava que els estalvis dels rics, si augmentaven amb la desigualtat, compensarien la reducció de la demanda dels consumidors. Un informe del 2013 sobre Nigèria suggereix que el creixement ha augmentat amb una desigualtat d’ingressos més gran. Algunes teories populars des de la dècada dels anys 50 fins al 2011 afirmaven, incorrectament, que la desigualtat tenia un efecte positiu en el desenvolupament econòmic. Les anàlisis basades en la comparació de les xifres anuals sobre igualtat amb els índexs de creixement anual van ser erronis ja que calen diversos anys perquè els efectes es manifestin com a canvis en el creixement econòmic. Els economistes de l’FMI van trobar una estreta relació entre nivells més baixos de desigualtat en els països en desenvolupament i períodes sostinguts de creixement econòmic. Els països en desenvolupament amb una elevada desigualtat han "aconseguit iniciar un creixement amb un índex elevat durant uns quants anys”, però "els períodes de creixement més llargs estan estretament relacionats amb més igualtat en la distribució de la renda".
----
Pregunta: Quan es van desenvolupar les teories que suggereixen que la desigualtat pot tenir un efecte positiu en el desenvolupament econòmic?
Resposta: dècada dels anys 70
----
Pregunta: Segons una anàlisi del 1955, què es pensava que compensaria els estalvis dels rics?
Resposta: la reducció de la demanda dels consumidors
----
Pregunta: Quant de temps es necessita perquè els efectes es manifestin com a canvis en el creixement econòmic?
Resposta: diversos anys
----
Pregunta: A què s’associen els períodes de creixement més llargs?
Resposta:"""

example = 'Algunes teories desenvolupades durant la dècada dels anys 70 van definir possibles vies a través de les quals la desigualtat pot tenir un efecte positiu en el desenvolupament econòmic. Un informe del 2013 sobre Nigèria suggereix que el creixement ha augmentat amb una desigualtat d’ingressos més gran. Algunes teories populars des de la dècada dels anys 50 fins al 2011 afirmaven, incorrectament, que la desigualtat tenia un efecte positiu en el desenvolupament econòmic. Les anàlisis basades en la comparació de les xifres anuals sobre igualtat amb els índexs de creixement anual van ser erronis ja que calen diversos anys perquè els efectes es manifestin com a canvis en el creixement econòmic. Els economistes de l’FMI van trobar una estreta relació entre nivells més baixos de desigualtat en els països en desenvolupament i períodes sostinguts de creixement econòmic. Els països en desenvolupament amb una elevada desigualtat han "aconseguit iniciar un creixement amb un índex elevat durant uns quants anys”, però "els períodes de creixement més llargs estan estretament relacionats amb més igualtat en la distribució de la renda.'
# example = 'Algunes teories desenvolupades durant la dècada dels anys 70 van definir possibles vies a través de les quals la desigualtat pot tenir un efecte positiu en el desenvolupament econòmic. Segons una anàlisi del 1955, es pensava que els estalvis dels rics, si augmentaven amb la desigualtat, compensarien la reducció de la demanda dels consumidors. Un informe del 2013 sobre Nigèria suggereix que el creixement ha augmentat amb una desigualtat d’ingressos més gran. Algunes teories populars des de la dècada dels anys 50 fins al 2011 afirmaven, incorrectament, que la desigualtat tenia un efecte positiu en el desenvolupament econòmic. Les anàlisis basades en la comparació de les xifres anuals sobre igualtat amb els índexs de creixement anual van ser erronis ja que calen diversos anys perquè els efectes es manifestin com a canvis en el creixement econòmic. Els economistes de l’FMI van trobar una estreta relació entre nivells més baixos de desigualtat en els països en desenvolupament i períodes sostinguts de creixement econòmic. Els països en desenvolupament amb una elevada desigualtat han "aconseguit iniciar un creixement amb un índex elevat durant uns quants anys”, però "els períodes de creixement més llargs estan estretament relacionats amb més igualtat en la distribució de la renda".'
translated_en = translate_to_english(example)
print(len(example))
print(len(translated_en))
print(translated_en[-100:])

# %%
