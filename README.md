# Detection of focal to bilateral tonic–clonic seizures using a connected shirt

Link to article: [Detection of focal to bilateral tonic–clonic seizures using a connected shirt](https://doi.org/10.1111/epi.18021).

By [Oumayma Gharbi](https://www.researchgate.net/profile/Oumayma-Gharbi)<sup>1,2</sup>, Yassine Lamrani<sup>1,2</sup>, Jérôme St-Jean<sup>1,2</sup> Amirhossein Jahani<sup>1,2</sup>, Dènahin Hinnoutondji Toffa<sup>1,2</sup>, TP Yen Tran<sup>2</sup>, Manon Robert<sup>2</sup>, Dang Khoa Nguyen<sup>1,2</sup> and Elie Bou Assi<sup>1,2</sup>.

<sup>1</sup> Department of Neurosciences, Université de Montréal, Montréal, QC, Canada.\
<sup>2</sup> Centre de Recherche du Centre Hospitalier de l'Université de Montréal (CRCHUM), Montréal, QC, Canada.

![ML pipeline](./Figures/Figure_2.png)

## Abstract

Objective: This study was undertaken to develop and evaluate a machine learning-based algorithm for the detection of focal to bilateral tonic–clonic seizures (FBTCS) using a novel multimodal connected shirt.
 
Methods: We prospectively recruited patients with epilepsy admitted to our epilepsy monitoring unit and asked them to wear the connected shirt while under simultaneous video-electroencephalographic monitoring. Electrocardiographic (ECG) and accelerometric (ACC) signals recorded with the connected shirt were used for the development of the seizure detection algorithm. First, we used a sliding window to extract linear and nonlinear features from both ECG and ACC signals. Then, we trained an extreme gradient boosting algorithm (XGBoost) to detect FBTCS according to seizure onset and offset annotated by three board-certified epileptologists. Finally, we applied a postprocessing step to regularize the classification output. A patientwise nested cross-validation was implemented to evaluate the performances in terms of sensitivity, false alarm rate (FAR), time in false warning (TiW), detection latency, and receiver operating characteristic area under the curve (ROC-AUC). 

Results: We recorded 66 FBTCS from 42 patients who wore the connected shirt for a total of 8067 continuous hours. The XGBoost algorithm reached a sensitivity of 84.8% (56/66 seizures), with a median FAR of .55/24h and a median TiW of 10 s alarm. ROC-AUC was .90 (95% confidence interval = .88–.91). Median detection latency from the time of progression to the bilateral tonic–clonic phase was 25.5 s. 

Significance: The novel connected shirt allowed accurate detection of FBTCS with a low false alarm rate in a hospital setting. Prospective studies in a residential setting with a real-time and online seizure detection algorithm are required to validate the performance and usability of this device.

## Information

This repository is for reference purpose only.

## Contact

For any questions, please contact the corresponding author.

## References

```bibtex
@article{gharbi2024,
  author = {Gharbi, Oumayma and Lamrani, Yassine and St-Jean, Jérôme and Jahani, Amirhossein and Toffa, Dènahin Hinnoutondji and Tran, Thi Phuoc Yen and Robert, Manon and Nguyen, Dang Khoa and Bou Assi, Elie},
  title = {Detection of focal to bilateral tonic–clonic seizures using a connected shirt},
  journal = {Epilepsia},
  date = {2024-05-23},
  volume = {n/a},
  number = {n/a},
  pages = {},
  keywords = {acceleration, biomedical signal processing, connected shirt, electrocardiogram, epilepsy, machine learning, seizure detection, wearable biosensors},
  doi = {https://doi.org/10.1111/epi.18021},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/epi.18021},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/epi.18021},
}
```

Gharbi, O., Lamrani, Y., St-Jean, J., Jahani, A., Toffa, D. H., Tran, T. P. Y., Robert, M., Nguyen, D. K., & Bou Assi, E. (2024). Detection of focal to bilateral tonic-clonic seizures using a connected shirt. Epilepsia, 10.1111/epi.18021. Advance online publication. https://doi.org/10.1111/epi.18021

This is an open access article under the terms of the Creative Commons Attribution-NonCommercial License, which permits use, distribution and reproduction in any 
medium, provided the original work is properly cited and is not used for commercial purposes.

&copy; 2024 The Authors. Epilepsia published by Wiley Periodicals LLC on behalf of International League Against Epilepsy.
