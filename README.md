# MFMSR:Multi-Scale Feature Aggregation Mamba for Remote Sensing Image Super-Resolution
In recent years, conventional neural networks such as CNNs and Transformers have achieved
substantial progress in remote sensing image super-resolution. Nevertheless, the exploration of
Mamba networks in this domain remains limited. Directly applying Mamba to remote sensing
images, which are characterized by abundant details and distinct structural patterns, may result in
the loss of fine-grained information. To mitigate this limitation, this study introduces a Global–Local
Feature Aggregation Mamba Block (FAMB). By expanding and evenly partitioning channels for
local and global processing, FAMB enables Mamba to perform comprehensive global modeling
while effectively enhancing local detail representation. Furthermore, to achieve robust Multi-Scale
feature aggregation in remote sensing imagery, we propose a Multi-Scale Vision State Space Model
(MVSSM) together with a Frequency Domain Channel Attention (FDCA) module. MVSSM enhances
the model’s capability to perceive objects of various scales and improves the stability of macroscopic
structure modeling through feature scanning at multiple scales. FDCA employs the 2D Discrete
Cosine Transform (2D DCT) to decompose frequency components, thereby explicitly distinguishing
information across different frequency bands. This mechanism emphasizes informative frequency
regions while suppressing noise components, effectively alleviating artifacts and enhancing the overall
reconstruction quality. Extensive experiments conducted on three benchmark datasets, namely AID,
DOTA, and DIOR, demonstrate that the proposed MFMSR framework effectively captures both
structural and detailed information in remote sensing images. In terms of PSNR, MFMSR outperforms
the Mamba-based MambaIR method by an average margin of 0.10 dB, and surpasses the remote
sensing super-resolution method FMSR by an average of 0.06 dB, simultaneously achieving a 7.04%
reduction in parameter count and a 26.81% decrease in memory consumption compared to MambaIR.
