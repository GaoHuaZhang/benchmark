分析 Dataset 和 Evaluator 组件的接口。查看相关代码：


[5 tools called]


查看更多实现示例以完善接口描述：


[4 tools called]


## Dataset 组件接口文档

### 一、接口描述

Dataset 组件负责数据集的加载、预处理和管理。所有数据集类继承自 `BaseDataset`，通过注册机制集成到评测系统。

### 二、接口信息模型

#### 2.1 基类接口定义

```python
class BaseDataset:
    """数据集基类"""

    def __init__(
        self,
        reader_cfg: Optional[Dict] = {},      # 数据集读取器配置
        k: Union[int, List[int]] = 1,        # 重复因子k（用于pass@k计算）
        n: int = 1,                           # 重复次数n（用于多次独立推理）
        abbr: str = 'dataset',                # 数据集唯一标识
        **kwargs                              # 其他参数（如path等）
    )

    @abstractmethod
    @staticmethod
    def load(**kwargs) -> Union[Dataset, DatasetDict]:
        """加载数据集（抽象方法，必须实现）"""
        pass

    @property
    def train(self) -> Dataset:
        """训练集属性"""

    @property
    def test(self) -> Dataset:
        """测试集属性"""
```

#### 2.2 接口数据模型

**输入参数：**
- `reader_cfg` (Dict): 数据集读取器配置
  - `test_range` (str): 测试范围，如 `"[0:100]"` 或 `"[0:8]"`
  - `output_column` (str): 输出列名（标准答案列）
  - 其他读取器特定参数

- `k` (Union[int, List[int]]): 重复因子
  - 用于计算 pass@k、cons@k 等指标
  - 可以是单个整数或整数列表

- `n` (int): 重复次数
  - 用于多次独立推理场景
  - 必须满足：max(k) <= n

- `abbr` (str): 数据集唯一标识
  - 用于区分不同数据集任务
  - 格式建议：小写字母和短横线组合

- `**kwargs`: 其他参数
  - `path` (str): 数据集路径（必需）
  - 其他数据集特定参数

**输出数据：**
- `self.dataset` (Union[Dataset, DatasetDict]): HuggingFace Dataset 对象
- `self.reader` (DatasetReader): 数据集读取器实例
- `self.abbr` (str): 数据集标识

### 三、接口清单

#### 3.1 核心接口

| 接口名称 | 类型 | 说明 | 是否必需 |
|---------|------|------|---------|
| `load(**kwargs)` | 静态方法 | 加载数据集，返回 Dataset 或 DatasetDict | 是（抽象方法） |
| `__init__(reader_cfg, k, n, abbr, **kwargs)` | 构造方法 | 初始化数据集实例 | 是 |
| `train` | 属性 | 获取训练集 | 否（可选） |
| `test` | 属性 | 获取测试集 | 否（可选） |

#### 3.2 注册接口

```python
@LOAD_DATASET.register_module()
class YourDataset(BaseDataset):
    @staticmethod
    def load(**kwargs):
        # 实现数据集加载逻辑
        pass
```

#### 3.3 实现示例

**示例1：GSM8K数据集**
```python
@LOAD_DATASET.register_module()
class GSM8KDataset(BaseDataset):
    @staticmethod
    def load(path):
        # 从JSONL文件加载数据
        datasets = {}
        for split in ['train', 'test']:
            split_path = os.path.join(path, split + '.jsonl')
            dataset = []
            with open(split_path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
            datasets[split] = Dataset.from_list(dataset)
        return DatasetDict(datasets)
```

**示例2：自定义数据集**
```python
@LOAD_DATASET.register_module()
class CustomDataset(BaseDataset):
    @staticmethod
    def load(path, file_name=None, meta_path='', local_mode=False):
        # 支持JSONL和CSV格式
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                data = [json.loads(line) for line in f]
        elif path.endswith('.csv'):
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader)
                data = [dict(zip(header, row)) for row in reader]
        return Dataset.from_list(data)
```

---

## Evaluator 组件接口文档

### 一、接口描述

Evaluator 组件负责评估模型预测结果与标准答案的匹配度，计算精度指标。所有评估器继承自 `BaseEvaluator`，通过注册机制集成到评测系统。

### 二、接口信息模型

#### 2.1 基类接口定义

```python
class BaseEvaluator:
    """评估器基类"""

    def __init__(self) -> None:
        """初始化评估器"""

    def evaluate(
        self,
        k: Union[int, List[int]],      # Top k值列表
        n: int,                         # 重复次数
        original_dataset: Dataset,     # 原始数据集
        **score_kwargs                 # 评分参数（predictions, references等）
    ) -> Dict[str, Any]:
        """评估主入口（已实现）"""

    @abstractmethod
    def score(
        self,
        predictions: List,              # 预测结果列表
        references: List,               # 标准答案列表
        **kwargs                        # 其他参数
    ) -> Dict[str, Any]:
        """计算评分（抽象方法，必须实现）"""
        pass

    def group(
        self,
        n: int,                         # 重复次数
        details: List[Dict],            # 评估详情列表
        test_set: Dataset               # 测试集
    ) -> Dict[str, Any]:
        """按样本分组（已实现）"""

    def reduce(
        self,
        details: List[Dict],            # 评估详情列表
        k_list: List[int],              # k值列表
        n_val: int                      # 重复次数
    ) -> Dict[str, Any]:
        """聚合结果（已实现）"""

    def pred_postprocess(
        self,
        predictions: List
    ) -> Dict:
        """预测结果后处理（已实现）"""
```

#### 2.2 接口数据模型

**输入参数：**

`evaluate()` 方法参数：
- `k` (Union[int, List[int]]): Top k 值，用于计算 pass@k
- `n` (int): 重复次数，用于多次独立推理
- `original_dataset` (Dataset): 原始测试数据集
- `score_kwargs` (Dict): 包含以下键值
  - `predictions` (List): 模型预测结果列表
  - `references` (List): 标准答案列表
  - 其他评估器特定参数

`score()` 方法参数：
- `predictions` (List): 预测结果列表，每个元素为字符串或其他类型
- `references` (List): 标准答案列表，与 predictions 长度相同
- `**kwargs`: 其他参数（如 `test_set` 等）

**输出数据：**

`score()` 方法返回：
```python
{
    'accuracy': float,          # 准确率（0-100）
    'details': List[Dict],       # 详细评估结果列表
    # 其他评估器特定指标
}
```

`evaluate()` 方法返回：
```python
{
    'accuracy': float,                    # 准确率
    'avg@{n}': float,                     # 平均准确率
    'pass@{k}': float,                    # pass@k指标
    'cons@{k}': float,                    # cons@k指标
    '{subdivision}/accuracy': float,      # 分类别准确率
    'details': List[Dict],                # 详细评估结果
    # 其他聚合后的指标
}
```

### 三、接口清单

#### 3.1 核心接口

| 接口名称 | 类型 | 说明 | 是否必需 |
|---------|------|------|---------|
| `score(predictions, references, **kwargs)` | 抽象方法 | 计算评分，必须实现 | 是 |
| `evaluate(k, n, original_dataset, **score_kwargs)` | 实例方法 | 评估主入口，已实现 | 否（继承） |
| `group(n, details, test_set)` | 实例方法 | 按样本分组，已实现 | 否（继承） |
| `reduce(details, k_list, n_val)` | 实例方法 | 聚合结果，已实现 | 否（继承） |
| `pred_postprocess(predictions)` | 实例方法 | 预测后处理，已实现 | 否（继承） |

#### 3.2 注册接口

```python
@ICL_EVALUATORS.register_module()
class YourEvaluator(BaseEvaluator):
    def score(self, predictions: List, references: List, **kwargs) -> dict:
        # 实现评分逻辑
        return {
            'accuracy': accuracy_value,
            'details': details_list
        }
```

#### 3.3 实现示例

**示例1：准确率评估器（HuggingFace）**
```python
@ICL_EVALUATORS.register_module()
class AccEvaluator(HuggingfaceEvaluator):
    def __init__(self):
        super().__init__(metric='accuracy')

    def score(self, predictions: List, references: List) -> dict:
        # 使用HuggingFace evaluate库计算准确率
        metric = evaluate.load('accuracy')
        scores = metric.compute(
            predictions=preds,
            references=golds
        )
        return {'accuracy': scores['accuracy'] * 100}
```

**示例2：数学评估器（自定义逻辑）**
```python
@ICL_EVALUATORS.register_module()
class MATHEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        correct = 0
        count = 0
        details = []
        for pred, ref in zip(predictions, references):
            count += 1
            # 解析和验证数学表达式
            gold_parsed = parse(ref, ...)
            answer_parsed = parse(pred, ...)
            answer_correct = float(verify(answer_parsed, gold_parsed))
            correct += answer_correct
            details.append({
                'pred': str(answer_parsed),
                'answer': str(gold_parsed),
                'correct': bool(answer_correct)
            })
        return {
            'accuracy': 100 * correct / count,
            'details': details
        }
```

**示例3：GSM8K评估器（数值比较）**
```python
class Gsm8kEvaluator(BaseEvaluator):
    def is_equal(self, pred, refer):
        """判断预测值与标准答案是否相等"""
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        return False

    def score(self, predictions, references):
        correct = 0
        count = 0
        details = []
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            count += 1
            if self.is_equal(pred, ref):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        return {
            'accuracy': 100 * correct / count,
            'details': details
        }
```

### 四、接口使用流程

#### Dataset 使用流程：
```
1. 定义数据集类，继承 BaseDataset
2. 实现 load() 静态方法
3. 使用 @LOAD_DATASET.register_module() 注册
4. 在配置文件中引用数据集
```

#### Evaluator 使用流程：
```
1. 定义评估器类，继承 BaseEvaluator
2. 实现 score() 方法
3. 使用 @ICL_EVALUATORS.register_module() 注册
4. 在数据集配置的 eval_cfg 中指定评估器
```

### 五、接口约束与注意事项

**Dataset 约束：**
- `load()` 必须返回 `Dataset` 或 `DatasetDict`
- `k` 的最大值不能超过 `n`
- 数据集路径必须可访问
- 支持的数据格式：JSONL、CSV、JSON 等

**Evaluator 约束：**
- `score()` 必须返回包含 `accuracy` 和 `details` 的字典
- `predictions` 和 `references` 长度必须相同
- `details` 中应包含每条样本的评估详情
- 支持多次独立推理（n > 1）时，需在 `details` 中包含 `correct` 或 `is_correct` 字段

以上是 Dataset 和 Evaluator 组件的接口文档。