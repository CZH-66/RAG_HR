## 已知检索失败案例（口语化测试集）

1. Q: "can I start at 8am?"
   预期: 关于 flexible start times 的政策
   实际: 检索到 [片段摘要]，未命中目标记录
   诊断: 用户问法省略了 "flexible working hours" 关键词
   计划: 评估是否引入查询改写

[9/12] hit=False  sim=0.814  how long do I have to file my expense claim before...
[10/12] hit=False  sim=0.755  my colleague and I keep butting heads, honestly ge...
