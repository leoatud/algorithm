binary tree path sumIII
a/b > c/d ==> a*d > b*c

backtracking --> DFS

breadth first search
宽度优先搜索
BFS in Graph : topological sorting

When using BFS:
	Travesal in Graph(层级遍历、点到面、拓扑)
	Shortest path in simple Graph
		最短路径算法:--google

UndirectedGraph
简单图：BSF fastest

BFS in binaryTree:

	Queue : interface : 未定义具体实现--> normally linkedlist
	Queue<TreeNode> queue = new LinkedList<>();
	环形数组：节省空间-->循环数组
	空间不够--> resize() ： rotatedsortedarray
	//dynamic array/ circular array/ arrayDeque/ interface
	BFS(QUEUE) 


	Queue<TreeNode> queue = new LinkedList<>();
	quese.offer(root);
	while(!queue.isEmpty()){
		List<Integer> level = new ArrayList<>();
		int size = queue.size();
		for(int i=0;i<size;i++){
			TreeNode node = queue.poll();
			if(node.left!=null)
				queue.offer(node.left);
			if(node.right!=null)
				queue.offer(node.right);
		}
		result.add(level);
	}
	return result;


	Queue: first in first : offer进去后，poll出来
	Stack: first in last out: push进去，pop出来
	Queue : java中为interface，需要implement具体的like LinkedList

	DFS:
		if(root==null) return;
		if(height >= res.size()){
			res.add(new LinkedList<Integer>);
		}
		res.get(height).add(root.val);
		helper(res, root.left,height+1);
		helper(res,root.right,height+1);


Serialization(序列化)：
	内存中结构化数据-->字符串的过程
	object-->String
	string --> object (要可以反序列化)
	目的：数据持久化存储
	disk中无内存中的数据结构
	xml/ Json(hashMap)/ Thrift/ ProtoBuf
 

DFS:all you need is to overload/ helper
list.add(index,list);
转换print顺序的时候，可以直接选择插入顺序，
LinkedList: addFirst

if(node==null)
	continue: 
//用来处理万一是null的，因为null也需要在打印中

Graph VS Tree:
环， 无环
连接表->基本数据结构定义
Map<Integer,Set<Integer>> graph = initializeGraph(n, edges);
valid graph tree:
	验证点和边的个数：edges.length = n-1
	从o出发可以访问到所有的节点
BFS:通过一个点可以访问到所有的点即可，对比数量

clone graph: deep copy
;




//list 往前implement方式
while(){
	list.val = sum/10;
	ListNode head = new ListNode(sum%10);
	head.next = list;
	list = head;
}


//解决倒数问题：双针、递归 in ListNode里面
//listNode always consider the head: if head ==null
因此一般在head前加一个pre:
ListNode pre = new ListNode(-1);
pre.next=head;

finally: return pre.next;

用过一个点找所有的点：bfs
主干先去实现-->先写注释/ 能分开的尽量分开、代码逻辑

BFS:::
Queue<UndirectedGraphNode> queue = new LinkedList<>();
HashSet<UndirectedGraphNode> set = new HashSet<>();
queue.offer(node);
set.add(node); //repeat cut
while(!queue.isEmpty()){
	UndirectedGraphNode head = queue.poll();
	for(UndirectedGraphNode nei: head.neighbors){
		if(!set.contains(nei)){
			set.add(nei);
			queue.offer(nei);
		}
	}
}
return new ArrayList<UndirectedGraphNode>(set);

能够用BFS就不用DFS： （用最简单理解的方式）
分层：加一个queue size(): for (i<size()){}

Topological Sorting: 拓扑排序==BFS(选课问题，课程之间依赖关系)
1. indegree	入度 （逐级删 除入度为0的点==修课）
2. outdegree 出读
面试先把接口写好，然后再具体implement 速度快，逻辑清晰，长一点没关系

//等于零的是没有重复的，相当于没有先修课的，其他都是被指向的
//graph起始可能有很多？！
o(M): M为边长：最深层

BSF in matrix: 
坐标变换数组
直接用matrix作为hashtable存储flag：是否访问过
一般三层循环：
while(!queue.isEmpty())
	int size = queue.size();
	for(int i=0;i<size;i++){  //分层原因
		node = queue.poll();
		for(int i=0;i<n;i++){  //matrix原因
			//将复杂的if换成一个interface
		}
	}            

大写字母：常量:通过与常量对比
增加易读性

上课问题：若两个课有环互相指向，可以两门课同时上
1： 找到度数为零的（不需要其他的）
2： 找环：
3：遍历模型： 贪心策略：找degree为0的课程上
































DFS：找所有方案、排列、组合 (stackoverflow)
： Recursion： 定义、拆解、出口
	节省空间：不需要保存每一个状态，只需要存储stack深度 
non recursion DFS: using Stack

//** from jiuzhang
搜索的时间复杂度：O(答案总数 * 构造每个答案的时间)
举例：Subsets问题，求所有的子集。子集个数一共 2^n，每个集合的平均长度是 O(n) 的，所以时间复杂度为 O(n * 2^n)，同理 Permutations 问题的时间复杂度为：O(n * n!)

动态规划的时间复杂度：O(状态总数 * 计算每个状态的时间复杂度)
举例：triangle，数字三角形的最短路径，状态总数约 O(n^2) 个，计算每个状态的时间复杂度为 O(1)——就是求一下 min。所以总的时间复杂度为 O(n^2)

用分治法解决二叉树问题的时间复杂度：O(二叉树节点个数 * 每个节点的计算时间)
举例：二叉树最大深度。二叉树节点个数为 N，每个节点上的计算时间为 O(1)。总的时间复杂度为 O(N)
//** end from jiuzhang

graph一般都不用DFS-->是一种算法
DFS：可以找所有路径、所有方案：
	backtracking 是DFS中的一个step


if(target == 0){
	result.add(new ArrayList<Integer>(combination));
	return;
}

for(int i=startIndex;i<nums.length;i++){
	if(combination[i]>target) break;
	combination.add(nums[i]);
	helper(nums,i,nums,target-combination[i],result);
	combination.remove(nums.size()-1); //cut repeat
}

加入后又remove要理解清楚：
去重：
if(i!=startIndex && candidates[i]==candidates[i-1])
	continue;
if(target < candidates[i])
	break;

先添加再删除的方式-->backtracking
return all possible: 一般都是DFS  
核心：加一个-->递归-->挪开
part.add(subString);
helper(0,i+1,part,result);
part.remove(size()-1);
出口：当前length==总length

排列类：O(n!)

思考方式：按照BFS：但是执行：
if(size1() == size2())
	result.add();
	return;

for(){
	if() continue;

	part.add();
	helper();  //dfs
	part.remove();
}

word ladder: BFS, graph最短路径
hash表：O(L) : L为hashtable里面string的长度
word ladder2: BFS+DFS
字符串最短变化序列 

不backtracking的dfs:不关心路径，只关心结果的
找path而非找point： 

BFS：point
DFS: path




















LinkedList & Array

 head==node1-->node2-->node3-->null : 别忘了Null
 node1 noden --> ref : 内容不变，只是更改ref了
 每个ref （4个字节）=4byte  == pointer (32位机)
 类似C里面的pointer

 Dummy Node very important
 double pointer solve reverse linkedlist problem
 改变链表结构：必须用dot
 temp = cur.next; //保存后面一位
 cur.next = pre; //后面指向前面

 pre = cur; //前面变成cur
 cur = temp;  //指针后移
	//整体往后挪
多开局部变量ListNode: ref： 逻辑清晰

ListNode reverseNextK(ListNode head, int k){
	ListNode n1 = head.next;
	ListNode nk = head;
	for(int i =0;i<k;i++){
		nk = nk.next;
		if(nk==null) return null;
	}
	//nk变动成最后一位

	ListNode nkplus = nk.next;
	ListNode pre = null;
	ListNode cur = n1;
	while(cur!=nkplus){
		ListNode temp = cur.next;
		cur.next = pre;
		pre = cur;
		cur = temp; //往后挪了一位
	}

	head.next = nk;
	n1.next = nkplus; 
	//n1为局部开头变脸，n1.next = nk.next；
	return n1;
}


如何使用dummy node: 
(当结构发生变化时)
	ListNode dummy = new ListNode(0);
	dummy.next = head;
	head = dummy;

Link cycle list:
	faster pointer vs slow pointer -->encounter

heap sort
bucket sort
redis sort

主要考：     time    space
quick sort  nlogn   o1
merge sort  nlogn	on
heap sort 	nlogn	o1

merge sort: recursive & divide conquer
	mergesort(arr){
		mergesort(arr.left half);
		mergesort(arr.right half);
		merge left & right in sorted order
	}
heap sort:
	heap = ordered binary tree
	parent > child

子数组：subarray --> prefixSum[i]  

median:::log级别
	-->findKth()  
	-->findKth(A, B, (n+m)/2);
	对比两个数组第Kth个数



















Two Pointer:
同向  & 异向
:时间复杂度：O(n)
partition array == quick sort
用while --> int left right -->永不回头

while(left<right){

	while(left<right && ...)

	while(left<right && ...)

	if(left<right){

	}
}

finally: if() return..;
		else return..;
判断条件统一写：while里面的统一写，因为涉及到操作
Quick Select : 快速排序、 非稳定排序，不保存原来内容
左中右三个部分的问题：
	分开问题：分两次就好

	三元指针： L, R, M

RainBow sort: quick sort:
	first partition
	second recursive

(merge sort different, first recursive and then partition)

坑爹排序：
	烙饼排序 pancake sort
	睡眠排序 sleep sort (通过thread wait相应时间输出)，坑爹
	面条排序
	猴子排序 bogo sort

去重：在while里面

非等于的：：：牛逼 
        while(left<right){
            if(nums[left]+nums[right]<=target){
                count += right-left;
                left++;
            }else{
                right--;
            }
        }
        return count;



 









Heap:
	priority Queue
	replacement: TreeMap
	data structure: 集合：
		在集合上面的若干操作

	Queue: O(1)push/ pop/ top 
		BFS: 

	Stack:O(1) push/pop/top 
		非递归实现DFS：

	Hash: O(1) insert/ find/ delete
		hashtable/ hashmap/ HashSet
		hash function : MD5/ SHA-1/ SHA-2 -->加密算法

magic number: 31 -->质数 ： apache底层：33
open hashing VS closed hashing : 再好的hash函数也会存在collision
hash不够大：-->rehash

LRU Cache: LinkedHashMap = DoublyLinkedList + hashMap

TreeMap: 又想知道最小值，又想支持修改和删除

上层结构  VS 底层结构：

连续 VS 不连续 （内存结构）

哈希表：array实现

heap:二叉树 （treeNode 不连续）

datasturcture VS memory : 不同的角度

对char操作，想要显示print出来： char - '0' -->具体结果
字符串转整数操作：num = num *10+c-'0'; (个位十位)

ExpressExpand: stack find "]"

两个stack-->可以反向 two stack --> reverse

Queue = two stack

Stack = two Queue

iterator: 目的： (stack实现)
	读文件：一行一行读，减少中间出错：分批load内存
	hasNext() next()
	主程序：在hasNext()里面实现

要么都在left解决，
要么都在right解决 -->因为是单头的

单调stack

for(it=new Iterator();it.hasNext();it.next()):
forloop based on this:

链表一般都是用While loop 
 
hashTable： safe thread (多线程访问时不会崩掉)
hash function:固定的，无规律的
mapping:多对一； key的对应value唯一


int hashFunc(String key){
	int sum=0;
	for(int i=0;i<key.length();i++){
		sum = sum * 31 +(int)(key.charAt(i));
		sum = sum % Hash_Table_Size; //capacity
		//变成数组下标
	}
	return sum;
}

closed hashing: 若被占用，占后面一个
open hashing: 占坑排队-->就要这个坑 （现实用的比较多）
		openning里面每个坑都是一个linkedlist

hash table size >= 1/10 --> （slow）rehashing --> *2 double size
rehashing: hashtable:add lock（during rehashing）
hash:耗费内存 -->牺牲空间增加速度
Tire: 前置树（hash升级）
LRU： 淘汰算法 least recently used 
	cache hit 
	cache miss
	线性时间轴：链式：linkedlist
	dummy node： 头删除快
HashMap<key, DoubleListNode>
单向链表okay：比较复杂

PriorityQueue(pq:排好序的)  VS Heap（minHeap / maxHeap）
java: from litte to big 
c++: from big to small
	delete: O(n)
BBST: balanced binary search tree:Olog(n)

数据结构会的越多，算法能力需求越弱

heap用法：在一个系列里面找最大或者最小

















递归+动规 +++ 记忆化搜索
Divide & Conquer + Memorization
记忆化搜索的本质

自底向上 VS 自顶向下
DP: 最大最小值、判断是否可行、统计方案个数
	只能记录一种最优的方案
	
DP：state + function + init + answer 
递归：define + divide + exit 

六种类型：坐标形DP、接龙型、划分型、双序列、背包型、区间型 

三角形题目--> DSF：变成二叉树

//找所有的路径：O(2^n) -->找所有路径DFS
void traverse(int x, int y, int sum){
	if(x==n){
		if(sum < best) best = sum;
		return;
	}
	traverse(x+1,y,sum+A[x][y]);
	traverse(x+1,y+1,sum+A[x][y]);
}


//divide & conquer O(2^n) --> 1 define 2 exit 3 conditions
int didiveConquer(int x, int y){
	if(x==n) return 0;
	return A[x][y] + Math.min(
		divideConquer(x+1,y),
		divideConquer(x+1,y+1)
	);
}

DFS-->优化 （冗余计算，重复计算）  记忆化搜索： DP：算法之上的算法
// add memorization --> O(n^2)
int didiveConquer(int x, int y){
	if(x==n) return 0;

	if(hash[x][y]!=Integer.MAX_VALUE)
		return hash[x][y];
	//save the step result
	hash[x][y] =  A[x][y] + Math.min(
		divideConquer(x+1,y),
		divideConquer(x+1,y+1)
	);
	return hash[x][y];
} 

DP：多重循环 VS 记忆化搜索
使用算法： O(2^n)/(n!)  --> O(n^2)/(n^3)  ->对暴力算法进行优化
1. 求最大，最小值
2.判断是否可行
3.统计方案个数

P问题：可以用多项式解决的问题（暴力算法）达到多项式级别
input为集合而非序列
具体方案，而非方案个数
===》》》 no DP

坐标型： 有起始点和重点: 移动时方向不能回头
		有坐标的概念
		2D DP--> init A[0][0] 初始化

	统计方案个数DP： unique path
	杨辉三角形
	f[i] = f[i-1] + f[i-2]
	climing stairs

	四个方向都可以走的：滑雪问题（从高到低）
		只要不是一直绕圈，有限制条件就行

	三维DP f[i][j][k] 走了k步 ：增加步数限制

接龙型（属于坐标型一种）：
	longest increasing subsequence
	只要我想出来的greedy方法一定是错的-->纯数学问题

	状态转移方程 + 初始化与答案

	int[] dp = new int[n+1];
	Arrays.fill(dp, Integer.MAX_VALUE);
	首先有初始状态array

	for(int i=0;i<=n;++i){
		for(int j=1;j*j<=i;++j){
			dp[i]=Math.min(dp[i],dp[i-j*j]+1);
		}
	}
	return dp[n];

	DP:无法记录所有最优方案，只能记录最最优的方案

	可以有两个记录数组： int pre[]: 倒着推断最优的结果全集

DP四要素：状态，方程，初始化，答案
	LIS 
































































