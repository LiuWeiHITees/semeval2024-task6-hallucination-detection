# semeval2024-task6-hallucination-detection

This paper describes our submission for SemEval-2024 Task 6: SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes, and we join both model-agnostic and model-aware tracks. We ensemble different methods, which significantly enhances the generalization capability. Our method's effectiveness is validated by our high rankings - 3rd in the model-agnostic track and 5th in the model-aware track.


## LLM-based Data Construction
Due to the lack of annotation data, it is difficult to incorporate task-oriented optimization for the pre-trained models. However, both tracks provide unannotated data with the form of [source, target, hypothesis]. Therefore, we propose to derive hallucination annotation by ourselves, leveraging the intelligence ability of proprietary LLMs, such as GPT-4. Based on the different tasks, we provide the paired text to the LLM, and design the prompt template to detect whether there is hallucination in the hypothesis. The specific dataset construction process is shown in Figure 1.

\begin{table*}[]
\resizebox{1.0\textwidth}{!}{
  \centering
    \begin{tabular}{lllrrrr}
    \specialrule{.1em}{.05em}{.05em}
    \multirow{2}[2]{*}{\textbf{Model Type}} & \multirow{2}[2]{*}{\textbf{Model}} & \multirow{2}[2]{*}{\textbf{Description}} & \multicolumn{2}{c}{\textbf{model-agnostic}} & \multicolumn{2}{c}{\textbf{model-aware}} \\
          &       &       & \multicolumn{1}{c}{acc} & \multicolumn{1}{c}{rho} & \multicolumn{1}{c}{acc} & \multicolumn{1}{c}{rho} \\
    \midrule
    Baseline & Mistral-7B & not train & 69.66 & 40.29 & 74.53 & 48.78 \\
    \midrule
    \multirow{3}[2]{*}{Entailment Model} & DeBERTa-MoritzLaurer & train and loss optimization & \textbf{82.46} & \textbf{75.20} & \textbf{80.46} & \textbf{71.23} \\
          & InternLM2 & train & 78.86 & 67.30 & 78.20 & 62.70 \\
          & InternLM2-sft & train & 63.53 & 50.35 & 64.86 & 46.77 \\
    \midrule
    \multirow{2}[2]{*}{Similarity Model} & SBERT & not train & 76.80 & 63.73 & 75.66 & 62.65 \\
          & UniEval & not train & 72.00 & 58.04 & 73.13 & 54.43 \\
    \midrule
    Feature Augmentation &       &       &       &       & 67.26 & 38.39$^\dagger$ \\
    \midrule
    Ensemble & ensemble & ensemble all model & \textbf{83.06} & \textbf{76.77} & 79.73 & 72.37$^\dagger$ \\
    \specialrule{.1em}{.05em}{.05em}
    \end{tabular}}
  \caption{Experimental results. \textsuperscript{†}Please note that there is a slight discrepancy with the results submitted in the leaderboard due to the inability to calculate the rho for the PG task, as the hallucination probability for 8 data points was null. For the purpose of this calculation, null probabilities were treated as zero, which may have a minor impact on the rho results.}
  \label{tab:result}
\end{table*}%
