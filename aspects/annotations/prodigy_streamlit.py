"""
Example of a Streamlit app for an interactive Prodigy dataset viewer that also lets you
run simple training experiments for NER and text classification.
Requires the Prodigy annotation tool to be installed: https://prodi.gy
See here for details on Streamlit: https://streamlit.io.
"""
import random

import pandas as pd
import spacy
import streamlit as st
from prodigy.components.db import connect
from prodigy.models.ner import EntityRecognizer, merge_spans, guess_batch_size
from prodigy.models.textcat import TextClassifier
from prodigy.util import split_evals
from spacy import displacy
from spacy.util import filter_spans, minibatch

SPACY_MODEL_NAMES = ["en_core_web_sm"]
EXC_FIELDS = ["meta", "priority", "score"]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
COLOR_ACCEPT = "#93eaa1"
COLOR_REJECT = "#ff8f8e"


def guess_dataset_type(first_eg):
    if "image" in first_eg:
        return "image"
    if "arc" in first_eg:
        return "dep"
    if "options" in first_eg or "label" in first_eg:
        return "textcat"
    if "spans" in first_eg:
        return "ner"
    return "other"


def get_answer_counts(examples):
    result = {"accept": 0, "reject": 0, "ignore": 0}
    for eg in examples:
        answer = eg.get("answer")
        if answer:
            result[answer] += 1
    return result


def format_label(label, answer="accept"):
    # Hack to use different colors for the label (by adding zero-width space)
    return f"{label}\u200B" if answer == "reject" else label


st.sidebar.title("Prodigy Data Explorer")
db = connect()
db_sets = db.datasets
placeholder = "Select dataset..."
dataset = st.sidebar.selectbox(f"Datasets ({len(db_sets)})", [placeholder] + db_sets)
if dataset != placeholder:
    examples = db.get_dataset(dataset)
    st.header(f"{dataset} ({len(examples)})")
    if not len(examples):
        st.markdown("_Empty dataset._")
    else:
        counts = get_answer_counts(examples)
        st.markdown(", ".join(f"**{c}** {a}" for a, c in counts.items()))
        dataset_types = ["ner", "textcat", "dep", "pos", "image", "other"]
        guessed_index = dataset_types.index(guess_dataset_type(examples[0]))
        set_type = st.sidebar.selectbox("Dataset type", dataset_types, guessed_index)
        fields = list(examples[0].keys())
        default_fields = [f for f in fields if f[0] != "_" and f not in EXC_FIELDS]
        task_fields = st.sidebar.multiselect("Visible fields", fields, default_fields)
        st.dataframe(pd.DataFrame(examples).filter(task_fields), height=500)

        if set_type in ["ner", "textcat"]:
            st.sidebar.header("Viewer options")
            purpose = "tokenization & training" if set_type == "ner" else "training"
            spacy_model_title = f"spaCy model for {purpose}"
            spacy_model = st.sidebar.selectbox(spacy_model_title, SPACY_MODEL_NAMES)
            st.sidebar.subheader("Training configuration")
            n_iter = st.sidebar.slider("Number of iterations", 1, 100, 5, 1)
            dropout = st.sidebar.slider("Dropout rate", 0.0, 1.0, 0.2, 0.05)
            eval_split_label = "% of examples held back for evaluation"
            eval_split = st.sidebar.slider(eval_split_label, 0.0, 1.0, 0.2, 0.05)

        if set_type == "ner":
            st.subheader("Named entity viewer")
            nlp = spacy.load(spacy_model)
            merged_examples = merge_spans(list(examples))
            all_labels = set()
            for eg in merged_examples:
                for span in eg["spans"]:
                    all_labels.add(span["label"])
            colors = {}
            for label in all_labels:
                colors[label] = COLOR_ACCEPT
                colors[format_label(label, "reject")] = COLOR_REJECT
            ner_example_i = st.selectbox(
                f"Merged examples ({len(merged_examples)})",
                range(len(merged_examples)),
                format_func=lambda i: merged_examples[int(i)]["text"][:400],
            )
            ner_example = merged_examples[int(ner_example_i)]
            doc = nlp.make_doc(ner_example["text"])
            ents = []
            for span in ner_example.get("spans", []):
                label = format_label(span["label"], span["answer"])
                ents.append(doc.char_span(span["start"], span["end"], label=label))
            doc.ents = filter_spans(ents)
            html = displacy.render(doc, style="ent", options={"colors": colors})
            html = html.replace("\n", " ")  # Newlines seem to mess with the rendering
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            show_ner_example_json = st.checkbox("Show JSON example")
            if show_ner_example_json:
                st.json(ner_example)

            st.subheader("Train a model (experimental)")
            no_missing = st.checkbox(
                "Data is gold-standard and contains no missing values", False
            )
            start_blank = st.checkbox("Start with blank NER model", True)
            if st.button("🚀 Start training"):
                if start_blank:
                    ner = nlp.create_pipe("ner")
                    if "ner" in nlp.pipe_names:
                        nlp.replace_pipe("ner", ner)
                    else:
                        nlp.add_pipe(ner)
                    ner.begin_training([])
                else:
                    ner = nlp.get_pipe("ner")
                for label in all_labels:
                    ner.add_label(label)
                random.shuffle(examples)
                train_examples, evals, eval_split = split_evals(
                    merged_examples, eval_split
                )
                st.success(
                    f"✅ Using **{len(train_examples)}** training examples "
                    f"and **{len(evals)}** evaluation examples with "
                    f"**{len(all_labels)}** label(s)"
                )
                annot_model = EntityRecognizer(
                    nlp, label=all_labels, no_missing=no_missing
                )
                batch_size = guess_batch_size(len(train_examples))
                baseline = annot_model.evaluate(evals)
                st.info(
                    f"ℹ️ **Baseline**\n**{baseline['right']:.0f}** right "
                    f"entities, **{baseline['wrong']:.0f}** wrong entities, "
                    f"**{baseline['unk']:.0f}** unkown entities, "
                    f"**{baseline['ents']:.0f}** total predicted, "
                    f"**{baseline['acc']:.2f}** accuracy"
                )
                progress = st.progress(0)
                results = []
                result_table = st.empty()
                best_acc = 0.0
                for i in range(n_iter):
                    random.shuffle(train_examples)
                    losses = annot_model.batch_train(
                        train_examples,
                        batch_size=batch_size,
                        drop=dropout,
                        beam_width=16,
                    )
                    stats = annot_model.evaluate(evals)
                    stats = {
                        "Right": stats["right"],
                        "Wrong": stats["wrong"],
                        "Unknown": stats["unk"],
                        "Predicted Ents": stats["ents"],
                        "Loss": losses["ner"],
                        "Accuracy": round(stats["acc"], 3),
                    }
                    best_acc = (
                        stats["Accuracy"] if stats["Accuracy"] > best_acc else best_acc
                    )

                    def highlight(v):
                        is_best = v != 0 and v == best_acc
                        return f"background: {'yellow' if is_best else 'white'}"

                    results.append(stats)
                    results_df = pd.DataFrame(results, dtype="float")
                    result_table.dataframe(results_df.style.applymap(highlight))
                    progress.progress(int((i + 1) / n_iter * 100))

        elif set_type == "textcat":
            st.subheader("Train a model (experimental)")
            exclusive = st.checkbox("Labels are mututally exclusive", False)
            if st.button("🚀 Start training"):
                nlp = spacy.load(spacy_model)
                examples = list(examples)
                all_labels = set()
                for eg in examples:
                    all_labels.update(eg.get("accelt", []))
                    if "label" in eg:
                        all_labels.add(eg["label"])
                textcat = nlp.create_pipe("textcat")
                for label in all_labels:
                    textcat.add_label(label)
                textcat.begin_training()
                nlp.add_pipe(textcat)
                random.shuffle(examples)
                train_examples, evals, eval_split = split_evals(examples, eval_split)
                st.success(
                    f"✅ Using **{len(train_examples)}** training examples "
                    f"and **{len(evals)}** evaluation examples with "
                    f"**{len(all_labels)}** label(s)"
                )
                annot_model = TextClassifier(
                    nlp,
                    all_labels,
                    low_data=len(train_examples) < 1000,
                    exclusive_classes=exclusive,
                )
                progress = st.progress(0)
                results = []
                result_table = st.empty()
                best_acc = 0.0
                for i in range(n_iter):
                    loss = 0.0
                    random.shuffle(train_examples)
                    for batch in minibatch(train_examples, size=10):
                        batch = list(batch)
                        loss += annot_model.update(batch, revise=False, drop=dropout)
                    with nlp.use_params(annot_model.optimizer.averages):
                        stats = annot_model.evaluate(evals)
                    stats = {
                        "Loss": loss,
                        "F-Score": stats["fscore"],
                        "Accuracy": round(stats["accuracy"], 3),
                    }
                    best_acc = (
                        stats["Accuracy"] if stats["Accuracy"] > best_acc else best_acc
                    )

                    def highlight(v):
                        is_best = v != 0 and v == best_acc
                        return f"background: {'yellow' if is_best else 'white'}"

                    results.append(stats)
                    results_df = pd.DataFrame(results, dtype="float").round(3)
                    result_table.dataframe(results_df.style.applymap(highlight))
                    progress.progress(int((i + 1) / n_iter * 100))
