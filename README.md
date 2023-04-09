# pandas_consistency_compare

比较DataFrame中C++计算的结果和python的差异

config使用json

## 配置

|Item|Expression|Parameter|
|---|---|---|
|format|需要比较的文件格式|"parquet"|
|path_of_cpp|c++产生的文件目录|path/to/cpp_res_dic|
|path_of_python|python产生的文件目录|path/to/python_res_dic|
|float_acc|浮点数的保留精度|uint|
|nan_handling|NaN的处理方式|"ignore","labeling"|
|error_compute|误差的计算方式|"SSE","MSE","SAE","MAE"|

## 补充

### nan_handling

`NaN_handling`用于处理当两个tabel中的元素一侧为NaN，另一侧有值的情况

- [1] `ignore` 表示直接忽略，不计入误差统计
- [2] `labeling` 忽略但会单独列出所有不一致的位置

如果选择`labeling`的方式，则会单独生成一个标准为`NaN_miss_match.csv`

### error_compute

`error_compute`指定误差的计算方式，可选的有

- [1] `SSE`(和方差、误差平方和):The sum of square error

- [2] `MSE`(均方差、方差):Mean square error

- [3] `SAE`(绝对误差和):The sum of absolute error

- [3] `MAE`(平均绝对误差):Mean absolute error

### 误差的统计格式

目录内所有文件名相同的数据有相同的列(顺序不要求),会产生`per_col`,`per_table`,`normal`三个文件

#### normal

|file_name|col_0|col_1|......|col_n|
|-|-|-|-|-|
|file_0|X|X|......|X|
|file_1|X|X|......|X|
|file_2|X|X|......|X|
|......|...|...|......|...|

#### per_col

对全部文件的同一列误差求和
|col_0|col_1|......|col_n|
|-|-|-|-|
|X|X|......|X|

#### per_table

对单个文件的全部列误差求和
|file_name|error|
|-|-|
|file_0|X|
|file_1|X|
|......|X|
|file_n|X|



