# 封存汇图层-Fan

`builders/storage.py` 建汇表（`_indtree/inputs/storage_hubs.csv`）读的四张栅格，原样取自：

> Fan J-L, Xiang X, Yao Y, Li K, Li Z, Wei S, Diao Y, Ju Z, Li X, Li X, Peng B, Ma J, Qi S, Zhang X.
> Dataset of CO2 geological storage potential and injection rate capacity in China based on fine grid technology.
> *Scientific Data* 12, 640 (2025). <https://doi.org/10.1038/s41597-025-04875-3>
>
> 数据集：Figshare <https://doi.org/10.6084/m9.figshare.27646707.v1>（2025-04-17 发布），许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
> 本目录的文件未作改动。论文正文的许可是 CC BY-NC-ND 4.0，不随本目录分发。

| 文件 | 内容 | 单位 | SHA-256 |
|---|---|---|---|
| `DSA-storage potential.tif` | 深部咸水层逐格封存潜力 | Mt CO2 | `a5fde055928c57a12f7bfafe64016a62ee49dc59ab599bcc459609d30cb8d730` |
| `DSA-injection-AVG-Recommended.tif` | 深部咸水层逐格注入能力，加权平均法（论文式 12，作者推荐） | Mt CO2/a | `b4a19670b520785fa54ad9cbd6954ab07775c036771fa6bdbd3a230f4d68858d` |
| `EOR-storage potential.tif` | 油田逐格封存潜力 | Mt CO2 | `da7ffb7d9fdc0a4b73e16d409aec9f639bdc2045c5def830715917d97fc48e31` |
| `EOR-injectionl.tif` | 油田逐格注入能力，取格内油田的最大值 | Mt CO2/a | `a0d35006c992225ba1b311170befc7b214572fbe4d386c96d5e1f90f100b801b` |

5 km × 5 km 网格，WGS 1984 Lambert Conformal Conic。咸水层每格设一个注入点，格值即该点的年注入能力（论文式 8）；单位见数据集
xlsx 的 About 页。数据集另有 `DSA injection-MAX.tif`（逐格最大值法）与县、省两级统计表，本仓库不用，未收入。

重建汇表：`python scripts/build_storage_inputs.py --offshore-basin-dilation-px 6`（海上封存体按盆地合并，89 个汇）；
2026-10-10 用这四个文件重建，与库里的 `storage_hubs.csv` 逐字节相同。
