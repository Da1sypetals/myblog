+++
date = '2025-03-01'
title = 'Lsm Tree 实现备注'
+++

Lsm Tree 是一种内存-磁盘的层级式数据结构，常用于实现写多读少的存储引擎。

# 组件
- 内存部分
- 磁盘部分
- WAL

# 总体

# 初始化
需要 init flush thread。flush thread 的工作流程:
1. 等待 flush 信号量被 notify,获取一个 compact 信号量资源
2. 启动一个 sstwriter,写入这个 memtable
   - 一个 memtable 对一个 sst
3. 等到写入 sst 写完之后,才进行:
   - 从 frozen memtables、frozen memtable sizes 里面删除这个 memtable
   - 从 wal 里面删除这个 memtable 对应的 wal
   - update manifest

# Try Freeze
如果当前大小 > freeze size 那么就 freeze;进一步如果所有 frozen memtable 大小之和 > flush threshold,那么就 set flush signal。

# 写操作
1. 写 memtable
2. 写 WAL
3. try freeze

# 内存部分

## Put
1. 添加到 memtable;
2. 更新 size。
   - size 不需要特别精确,只需要是一个大致的值即可。

## Delete
1. 添加一个 tomb 标记到 memtable

## Get
1. 从 active memtable 中获取
2. 从 new 到 old 遍历所有的 inactive memtable,获取。

# 磁盘部分

## compact 信号量
二元信号量。
- 需要 compact 的时候,添加资源
- compact thread 开始 compact 的时候,消耗资源。

## 初始化
如果 auto compact 开启,初始化的时候需要 init compact thread:

## Level
存储这个 level 所有文件对应的文件路径,装在 sst reader 里面

## Get (没有 delete, put)
从低到高,从新到旧,调用 sst 的 get 方法,获取 record。否则返回 none。

## Init Compact Thread
Compact thread:
1. 等待 compact 信号量
2. 依次查看每一层:如果这一层大小超过 threshold,就合并到下一层,否则就提前返回。

## Compact
以 L0 -> L1 为例:
从前到后遍历所有的 kv-pair,同时维护:
1. keys_outdated
   - 同一个 key,timetsamp 小于 oldest marker 的 kv pair 只需要保留一个。
   - keys_outdated 记录所有(出现过的,且 timestamp 小于 oldest marker)的 key
2. L1 sst size 每达到一定值就关闭当前 sst,新开一个新的 sst。
3. 更新 manifest。

## SST writer
配置 max block size。
- 每个 block 的开头一个 key 会添加到 index 中;
- 搜索这个 sst 的时候,会先对 index 进行二分查找;
- 在 block 之内采用线性搜索。

fpr,用于构建 bloom filter.

### 写入
1. 遍历所有的 kv pair:
   - userkey(不含 timestamp)添加到 bloom filter;
   - block 写入当前 kv;
   - 如果当前 block 大小超过 max block size,就开启一个新的 block,然后写入对应的 index(内存)
2. 将 index 和 bloom filter 写磁盘。

### SST reader 查找: Get(key, timestamp)
1. 查 bloom filter,如果不存在就返回。
2. 将 index 整个载入内存中,进行二分查找,得到对应 key-timestamp 所在的区间。如果 out of bounds 就返回。
3. 按照查找到的区间,读磁盘。

# MVCC

## key 排布问题

### struct Key
- bytes
- timestamp: u64

比较: key1 < key2:
- key1.bytes < key2.bytes (字典序);
- 或者: key1.bytes == key2.bytes,而且 key1.timestamp > key2.timestamp

### 为什么这样比较?
在进行查询 Get(userkey, timestamp) 的时候,我们需要的是:
- userkey 匹配
- timestamp 小于查询的 timestamp,且尽可能大

因此,我们将
- userkey 升序排序
- timestamp 降序排序

在搜索 memtable(skiplist)的时候,或者对 index 进行二分查找的时候,就可以:
1. 直接使用 lower_bound,查找大于等于自己的第一个元素
2. 如果 userkey 匹配,说明是 timestamp 小于当前 timestamp 的,timestamp 最大的记录,返回;
3. 如果 userkey 不匹配,说明不存在 timestamp 小于当前 timestamp 的记录,返回(未找到)。

# Transaction

## 数据结构
一个内存 tempmap,用来存储 transaction 已经写,但是未提交的内容。
创建的时候,从 tree 获取:
- start timestamp,作为查询的 timestamp
- transaction id

然后写入 transaction start 到 WAL

## Put,Delete
写 tempmap,写 WAL

## Get
使用 start timestamp,先查 tempmap,再查 tree。

## Commit
1. 从 tree 获取一个 commit timestamp;
2. 写 WAL,记录 transaction id 和 commit timestamp。
   - 在 replay 的时候,把 transaction id 和 commit timestamp 对应起来就可以知道 transaction 里面的 写操作 对应的 timestamp
3. 调用 tree.active_memtable 的 API,将 transaction 的所有数据写入 tree 的 memtable。

## WAL
看到 transaction start,先将 transaction 暂存到内存中:
- 如果在 replay 结束之前看到了 transaction end,就将改动写入 tree 中(redo)。
- 否则放弃,视为没完成的事务(undo)

# 踩坑:
1. Resource deadlock avoided (os error 35),可能是一个 thread 持有了自己的 joinhandle 并且 join 了自己;使用 maybe join 解决,即判断当前线程和 joinhandle 的线程是否一致,如果一致就不用 join。
2. 死锁问题: wal 和 mem 都有锁,必须 按照同一顺序获取 才不会出现死锁。

# Bloom filter 细节,by Deepseek
该 Bloom filter 算法的主要步骤如下:

1. 参数计算:
   - 根据预期元素数量 n 和可接受误判率 p,通过公式计算最优位数 m 和哈希函数数量 k:
     - m = ⌈-n·ln(p)/(ln2)^2⌉
     - k = ⌈(m/n)·ln2⌉
   - 当直接指定参数时,使用给定的位数和哈希函数数量

2. 哈希生成:
   - 使用 64 位指纹哈希(farmhash)生成初始哈希值 h
   - 通过位运算构造增量值 delta = (h >> 33) | (h << 31)
   - 采用双重哈希技术,通过循环叠加 delta 生成 k 个不同的位位置: h_i = h + i·delta (mod m)

3. 数据插入:
   - 对输入 key 进行哈希计算得到初始 h 和 delta
   - 循环 k 次生成位位置,将位数组中对应位置设为 1
   - 采用位操作: byte_index = position/8,bit_mask = 1 << (position%8)

4. 存在性检测:
   - 重复插入时的哈希计算过程
   - 检查所有 k 个对应位是否均为 1
   - 任一位置为 0 则判定不存在,全部为 1 时判定可能存在

5. 数据持久化:
   - 序列化时附加 CRC32 校验和
   - 反序列化时验证校验和与数据完整性