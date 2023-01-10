.. notebook_test documentation master file, created by
   sphinx-quickstart on Sat Jul 25 11:56:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎一起学习UVA深度学习教程!
===========================================

| *课程网站*: https://uvadlc.github.io/
| *Course edition*: DL1 - Fall 2022, DL2 - Spring 2022, Being kept up to date
| *notebook地址*: https://github.com/phlippe/uvadlc_notebooks
| *视频链接*: `YouTube Playlist <https://www.youtube.com/playlist?list=PLdlPlO1QhMiAkedeu0aJixfkknLRxk1nA>`_
| *作者*: Phillip Lippe
| *译者*: Jun Liang

.. note::
   有兴趣学习 JAX 吗？ 我们最近开始将笔记本从 PyTorch 翻译成 JAX+Flax。 在*深度学习 1 (JAX+Flax)* 选项卡中查看我们的新笔记本，了解如何使用 JAX 加速模型训练！


对于今年的课程版本，我们创建了一系列 Jupyter 笔记本，旨在帮助您通过查看相应的实现来理解讲座中的“理论”。
我们将涉及各种主题，例如优化技术、转换器、图形神经网络等（有关完整列表，请参见下文）。
这些笔记本可以帮助您理解材料并教您 PyTorch 框架的详细信息，包括 PyTorch Lightning。
此外，我们提供笔记本到 JAX+Flax 的一对一翻译作为可选框架。

笔记本在每个小组辅导课开始的前一小时教授。
在课程中，我们将展示内容并解释笔记本的实现。
您可以自己决定是只想看填满的笔记本、想自己尝试一下，还是在实践环节中一起编码。
这些笔记本不直接属于您将被评分或类似评分的任何强制性作业的一部分。
但是，我们鼓励您熟悉笔记本并自己进行实验或扩展。
此外，所呈现的内容将与评分作业和考试相关。

教程已整合为PyTorch Lightning官方教程。
因此，您还可以在 `他们的文档 <https://pytorch-lightning.readthedocs.io/en/latest/>`_ 中查看它们。

Schedule (Deep Learning 1, edition 2022)
----------------------------------------

+------------------------------------------+---------------------------------------------------------------+
| **Date**                                 | **Notebook**                                                  |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 1. November 2022, 17.00-18.00   | Tutorial 2: Introduction to PyTorch                           |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 8. November 2022, 17.00-18.00   | Tutorial 3: Activation functions                              |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 15. November 2022, 17.00-18.00  | Tutorial 4: Optimization and Initialization                   |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 22. November 2022, 17.00-18.00  | Tutorial 5: Inception, ResNet and DenseNet                    |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 29. November 2022, 17.00-18.00  | Tutorial 6: Transformers and Multi-Head Attention             |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 6. December 2022, 17.00-18.00   | Tutorial 7: Graph Neural Networks                             |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, 13. December 2022, 17.00-18.00  | Tutorial 17: Self-Supervised Contrastive Learning with SimCLR |
+------------------------------------------+---------------------------------------------------------------+

如何运行notebooks
-----------------------------------

在这个网站上，您会发现导出为 HTML 格式的笔记本，以便您可以从任何您喜欢的设备上阅读它们。
但是，我们建议您也尝试一下并自己运行它们。 我们推荐三种主要的笔记本运行方式：

- **本机 CPU**：所有笔记本都存储在 github 存储库中，该存储库也构建了这个网站。 您可以在这里找到它们：https://github.com/phlippe/uvadlc_notebooks/tree/master/docs/tutorial_notebooks。 笔记本电脑的设计让您无需 GPU 即可在普通笔记本电脑上执行它们。 我们提供在运行笔记本时自动下载的预训练模型，或者可以从这个 `Google Drive <https://drive.google.com/drive/folders/1SevzqrkhHPAifKEHo-gi7J-dVxifvs4c?usp=sharing>`_ 手动下载 . 预训练模型和数据集所需的磁盘空间小于 1GB。 为确保您安装了所有正确的 python 包，我们在 `相同的存储库 <https://github.com/uvadlc/uvadlc_practicals_2020/blob/master/environment.yml>`_ 中提供了一个 conda 环境（选择 CPU 或 GPU 版本取决于您的系统）。

- **Google Colab**：如果您更喜欢在与您自己的计算机不同的平台上运行笔记本，或者想要试验 GPU 支持，我们建议使用 `Google Colab <https://colab.research.google.com /notebooks/intro.ipynb#recent=true>`_ 。 本文档网站上的每个笔记本都有一个徽章，上面有一个链接，可以在 Google Colab 上打开它。 请记住在运行Notebook之前启用 GPU 支持（：代码：`Runtime -> Change runtime type`）。 每个Notebook都可以独立执行，不需要您连接 Google Drive 或类似设备。 但是，关闭会话时，如果您未将其保存到本地计算机或事先将笔记本复制到您的 Google 云端硬盘，则更改可能会丢失。

- **Lisa 集群**：如果您想基于笔记本训练自己的（更大的）神经网络，您可以使用 Lisa 集群。 但是，只有当你真的想训练一个新模型时才建议这样做，并使用其他两个选项来完成模型的讨论和分析。 Lisa 可能不允许您使用学生帐户直接在 gpu_shared 分区上运行 Jupyter 笔记本。 相反，您可以先使用 :code:`jupyter nbconvert --to script ...ipynb` 将笔记本转换为脚本，然后在 Lisa 上启动作业以运行该脚本。 在 Lisa 上跑步时的一些建议：

    - 禁用笔记本中的 tqdm 语句。 否则，您的 slurm 输出文件可能会溢出并且有几 MB 大。 在 PyTorch Lightning 中，您可以通过在训练器中设置：code:`enable_progress_bar=False` 来做到这一点。
    - 注释掉 matplotlib 绘图语句，或将 :code:`plt.show()` 更改为 :code:`plt.savefig(...)`。

教程-讲座对齐
--------------------------

我们将讨论课程中的 7 个教程，分布在各个讲座中以涵盖各个领域的内容。 您可以根据主题将教程与讲座对齐。 深度学习 1 课程中的教程列表是：

- 指南 1：使用 Lisa 集群
- 教程 2：PyTorch 简介
- 教程 3：激活函数
- 教程 4：优化和初始化
- 教程 5：Inception、ResNet 和 DenseNet
- 教程 6：Transformers 和 Multi-Head Attention
- 教程 7：图神经网络
- 教程 8：深度能量模型
- 教程 9：自动编码器
- 教程 10：对抗性攻击
- 教程 11：规范化图像建模流程
- 教程 12：自回归图像建模
- 教程 15：视觉Transformers
- 教程 16：元学习 - learning to learn
- 教程 17：使用 SimCLR 进行自监督对比学习

反馈、问题或贡献
----------------------------------

这是我们第一次在深度学习课程中介绍这些教程。 与任何其他项目一样，预计会出现小错误和问题。 我们感谢学生的任何反馈，无论是关于拼写错误、实施错误，还是对笔记本的改进/添加的建议。 请使用以下 `链接 <https://forms.gle/kENuNvcCq3LzQWDA8>`_  提交反馈，或随时通过邮件直接与我联系（p dot lippe at uva dot nl），或在任何 TA 期间联系我 会议。

如果您发现这些教程有帮助并想引用它们，您可以使用以下 bibtex::

   @misc{lippe2022uvadlc,
      title        = {{UvA Deep Learning Tutorials}},
      author       = {Phillip Lippe},
      year         = 2022,
      howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
   }


.. toctree::
   :caption: 指南
   :maxdepth: 2

   tutorial_notebooks/tutorial1/Lisa_Cluster
   tutorial_notebooks/guide2/Research_Projects
   tutorial_notebooks/guide3/Debugging_PyTorch
   tutorial_notebooks/guide4/Research_Projects_with_JAX

.. toctree::
   :caption: Deep Learning 1 (PyTorch)
   :maxdepth: 2

   tutorial_notebooks/tutorial2/Introduction_to_PyTorch
   tutorial_notebooks/tutorial3/Activation_Functions
   tutorial_notebooks/tutorial4/Optimization_and_Initialization
   tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet
   tutorial_notebooks/tutorial6/Transformers_and_MHAttention
   tutorial_notebooks/tutorial7/GNN_overview
   tutorial_notebooks/tutorial8/Deep_Energy_Models
   tutorial_notebooks/tutorial9/AE_CIFAR10
   tutorial_notebooks/tutorial10/Adversarial_Attacks
   tutorial_notebooks/tutorial11/NF_image_modeling
   tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling
   tutorial_notebooks/tutorial15/Vision_Transformer
   tutorial_notebooks/tutorial16/Meta_Learning
   tutorial_notebooks/tutorial17/SimCLR

.. toctree::
   :caption: Deep Learning 1 (JAX+Flax)
   :maxdepth: 2

   tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX
   tutorial_notebooks/JAX/tutorial3/Activation_Functions
   tutorial_notebooks/JAX/tutorial4/Optimization_and_Initialization
   tutorial_notebooks/JAX/tutorial5/Inception_ResNet_DenseNet
   tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention
   tutorial_notebooks/JAX/tutorial7/GNN_overview
   tutorial_notebooks/JAX/tutorial9/AE_CIFAR10
   tutorial_notebooks/JAX/tutorial11/NF_image_modeling
   tutorial_notebooks/JAX/tutorial12/Autoregressive_Image_Modeling
   tutorial_notebooks/JAX/tutorial15/Vision_Transformer
   tutorial_notebooks/JAX/tutorial17/SimCLR

.. toctree::
   :caption: Deep Learning 2
   :maxdepth: 2

   tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions
   tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns
   tutorial_notebooks/DL2/deep_probabilistic_models_I/tutorial_1.ipynb
   tutorial_notebooks/DL2/deep_probabilistic_models_II/tutorial_2a.ipynb
   tutorial_notebooks/DL2/deep_probabilistic_models_II/tutorial_2b.ipynb
   tutorial_notebooks/DL2/Advanced_Generative_Models/Normalizing_flows/advancednormflow.ipynb
   tutorial_notebooks/DL2/High-performant_DL/hyperparameter_search/hpdlhyperparam.ipynb
   tutorial_notebooks/DL2/High-performant_DL/Multi_GPU/hpdlmultigpu.ipynb
   tutorial_notebooks/DL2/Bayesian_Neural_Networks/Complete_DLII_BNN_2_2.ipynb
   tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_1.ipynb
   tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_2.ipynb
   tutorial_notebooks/DL2/Dynamical_systems/dynamical_systems_neural_odes.ipynb
   tutorial_notebooks/DL2/sampling/introduction.ipynb
   tutorial_notebooks/DL2/sampling/subsets.ipynb
   tutorial_notebooks/DL2/sampling/permutations.ipynb
   tutorial_notebooks/DL2/sampling/graphs.ipynb

