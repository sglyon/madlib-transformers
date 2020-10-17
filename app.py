import streamlit as st
import transformers
import spacy
import numpy as np
import math


# Tools


def const_hash(x):
    def inner(_foo):
        return x

    return inner


HASH_FUNCS = {
    transformers.pipelines.TextGenerationPipeline: const_hash("text-gen-model"),
    transformers.pipelines.FillMaskPipeline: const_hash("fill-mask-model"),
    "spacy.lang.en.English": const_hash("spacy-model"),
}


@st.cache(hash_funcs=HASH_FUNCS)
def gen_model():
    return transformers.pipeline("text-generation")


@st.cache(hash_funcs=HASH_FUNCS)
def gen_mask_filler():
    return transformers.pipeline("fill-mask")


@st.cache(hash_funcs=HASH_FUNCS)
def gen_spacy():
    spacy.prefer_gpu()
    return spacy.load("en_core_web_sm")


text_generator = gen_model()
spacy_model = gen_spacy()
mask_filler = gen_mask_filler()


@st.cache(hash_funcs=HASH_FUNCS)
def get_options(prompt):
    options = text_generator(
        prompt,
        max_length=70,
        top_p=0.9,
        top_k=30,
        temperature=0.5,
        num_return_sequences=3,
        do_sample=True,
        no_repeat_ngram_size=2,
    )
    return [x["generated_text"] for x in options]


def replace_pos_tokens(doc, pct, prompt):
    n_prompt = len(prompt.split())
    mask_token = mask_filler.tokenizer.mask_token

    # get index of tokens that we want madlibber to replace
    inds = []
    for pos in ["NOUN", "ADJ", "VERB"]:

        is_pos = np.array(list(map(lambda x: x.pos_ == pos, doc[n_prompt:]))) * 1.0
        n_pos = int(is_pos.sum())
        token_range = np.arange(n_prompt, len(doc))

        replace_inds = np.random.choice(
            token_range,
            size=min(math.floor(n_pos * pct) + 1, n_pos),
            replace=False,
            p=is_pos / is_pos.sum(),
        )
        inds.extend(list(replace_inds))

    # sort them so we can go through fully
    inds.sort()

    token_list = [tok.text for tok in doc]
    for ix in inds:
        before = token_list[ix]
        token_list[ix] = mask_token
        changes = mask_filler(" ".join(token_list))
        print(ix, before, "=>", changes[0]["token_str"][1:])
        token_list[ix] = changes[0]["token_str"][1:]

    for ix in inds:
        token_list[ix] = '<span style="color:red;">' + token_list[ix] + "</span>"
    return " ".join(token_list)


# app

st.title("Transformed MadLibs")

"""
In this application we will use Python NLP tools to perform various tasks:

1. The user can write a prompt or start of a story and the AI will complete the paragraph. This uses the awesome huggingface/transformers library and the `"text-generation"` pipeline
2. Spacy will be used to do part of speech tagging on whichever of the three "paragraph completions" the user selects.
3. huggingface/transformers will then be used in combination with Spacy's part of speech tagging to play madlibs! About 1/3 of all nouns, adjectives, and verbs will be masked and then filled in with the `"fill-mask"` pipeline

"""

"## Step 1: use text-generation pipeline on user supplied prompt"

prompt = st.text_input(
    "Write first few (~10) words of a story below",
    value="The big blue snake slithered past my feet",
)

f"Here is what you wrote: {prompt}"

options = get_options(prompt)

selected = st.radio("Select which continuation of the story you'd like to use", options)

quoted_selection = selected.replace("\n\n", "\n\n>")

f"You selected:\n\n > {quoted_selection}"

"## Step 2: Spacy part of speech tagging"

"Here is a view of the part of speech:"

doc = spacy_model(selected)

st.image(spacy.displacy.render(doc, style="dep", jupyter=False))

"## Step 3: Madlibs! fill-mask pipeline to replace nouns/adjectives/verbs"

output = replace_pos_tokens(doc, 1.0 / 3.0, prompt)

st.markdown(f"<div>{output}</div>", unsafe_allow_html=True)
