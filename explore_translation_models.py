# %%
import os
os.environ['XDG_CACHE'] = '/workspace/.cache'
os.environ['HF_HOME']='/workspace/.cache/huggingface'


# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

# %%
model_name = 'jbochi/madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name, padding_side='left')

# %%
def translate_to(txt, language_code):
    lines = [f"<2{language_code}> {l}" for l in txt.split("\n") if l.strip() != ""]
    toks = tokenizer(lines, return_tensors="pt", padding=True).input_ids.to(model.device)
    translated_toks = model.generate(input_ids=toks, max_new_tokens=2048)
    translated_lines = tokenizer.batch_decode(translated_toks, skip_special_tokens=True)
    return "\n".join(translated_lines)

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

#example = 'Algunes teories desenvolupades durant la dècada dels anys 70 van definir possibles vies a través de les quals la desigualtat pot tenir un efecte positiu en el desenvolupament econòmic. Un informe del 2013 sobre Nigèria suggereix que el creixement ha augmentat amb una desigualtat d’ingressos més gran. Algunes teories populars des de la dècada dels anys 50 fins al 2011 afirmaven, incorrectament, que la desigualtat tenia un efecte positiu en el desenvolupament econòmic. Les anàlisis basades en la comparació de les xifres anuals sobre igualtat amb els índexs de creixement anual van ser erronis ja que calen diversos anys perquè els efectes es manifestin com a canvis en el creixement econòmic. Els economistes de l’FMI van trobar una estreta relació entre nivells més baixos de desigualtat en els països en desenvolupament i períodes sostinguts de creixement econòmic. Els països en desenvolupament amb una elevada desigualtat han "aconseguit iniciar un creixement amb un índex elevat durant uns quants anys”, però "els períodes de creixement més llargs estan estretament relacionats amb més igualtat en la distribució de la renda.'
start = time.time()
translated_en = translate_to(example, "en")
print(translated_en)
print(f"Time: {time.time() - start}")
