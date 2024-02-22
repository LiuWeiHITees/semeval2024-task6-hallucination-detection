# semeval2024-task6-hallucination-detection

This paper describes our submission for SemEval-2024 Task 6: SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes, and we join both model-agnostic and model-aware tracks. We ensemble different methods, which significantly enhances the generalization capability. Our method's effectiveness is validated by our high rankings - 3rd in the model-agnostic track and 5th in the model-aware track.


## LLM-based Data Construction

The code in GPT-4-to-label-the-training-set.ipynb

Due to the lack of annotation data, it is difficult to incorporate task-oriented optimization for the pre-trained models. However, both tracks provide unannotated data with the form of [source, target, hypothesis]. Therefore, we propose to derive hallucination annotation by ourselves, leveraging the intelligence ability of proprietary LLMs, such as GPT-4. Based on the different tasks, we provide the paired text to the LLM, and design the prompt template to detect whether there is hallucination in the hypothesis.

## Main Results


\begin{itemize}
\item
  deberta: deberta-separately-for-each-task-train.py, deberta-loss-optimized-parameter-adjustment.py and deberta-loss-optimized-train.py.
\item
  sbert and unieval: sbert-and-unieval.ipynb
\item
  model aware feature augmentation: model-aware-feature-augmentation.ipynb
\item
  fine-tuning LLM: fine-tuning-LLM

\end{itemize}


![capture_20240222234101022](fig/result.bmp)
