# 修改code/readData.py内的内容，完成以下程序：
## 1.def readTrainData2File(num)

    传入参数num：为整形
    函数功能：	
    读取data/train_list里的txt文件，返回一个三维list[class][file][content]
      1. Class：类别，共9类
      2. File：class对应的所有的文件，排序按照list文件夹中的顺序生成
      3. Content:根据传入的num值，对file所指的文件进行随机抽取（不足num条时全部输出），抽取条目为num条，，抽取整行，将num条数据组成一个list，
      该list即为content（当num参数为0时，不随机抽取，返回全部内容）函数返回：将整个三维list输出到一个文件，数据结构自行设计，
       输出文件路径及命名为data/npy/train_num   num为传入的参数，文件后缀根据自己需要设计

## 2.def readTrainData2Mem()

  读取函数1生成的结果到内存，返回读取的list



## 3. def genTestList()

  参考data/train_list中的形式，将data/data/test_visit的所有文件按照顺序生成txt,存储为data/test_list/test.txt,具体形式翻聊天记录

## 3. def readTestData2File(num)
    
    与函数1功能相同，输出到文件为2维list[file][content]
    file: 因为此时没有分类，没有class维,此序列与test.txt内容序列一致
    content: 与函数1一致

## 4. def readTestData2Mem()

  读入数据



# 以上程序需要有必要的打印状态的功能
