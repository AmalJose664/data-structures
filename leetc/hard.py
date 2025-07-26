import collections
import sys
import os
import time

from helpers import ListNode, buildNodes, leetcode_output, print_array_with_pointers, print_matrix, showNodes
sys.path.append(os.path.dirname(__file__)) 

class HardSolution(object):
    
	def findMedianSortedArrays(self, nums1, nums2):
		# 4
		A,B = nums1,nums2
		total = len(nums1) + len(nums2)
		half = total //2
		
		if len(B)<len(A):
			A,B = B,A
		
		l, r = 0, len(A)-1
		while True:
			i = (l+r) //2
			j = half - i -2

			Aleft = A[i] if i>=0 else float('-infinity')
			Aright = A[i+1] if (i+1) < len(A) else float('infinity')
			Bleft = B[j] if j>=0 else float('-infinity')
			Bright = B[j+1] if (j+1) < len(B) else float('infinity')

			if Aleft <= Bright and Bleft<=Aright:
				if total % 2:
					return min(Aright, Bright)
				return (max(Aleft, Bleft)+ min(Aright, Bright)) /2
			elif Aleft > Bright:
				r = i-1
			else:
				l = i+1

	def isMatch(self, s, p):
		# 10
		print(s, p, '\n')
		cache = {}

		def dfs(i, j):
			print(f"Loop i = {i}, j = {j}")
			if (i, j) in cache:
				print("Data found return cache")
				return cache[(i, j)]
			if i >=len(s) and j>=len(p):
				print(f"         I out of bounds >>>>> {i} <<<<<")
				return True
			if j>=len(p):
				print(f"         J out of bounds >>>>> {j} <<<<<")
				return False
			
			match = i < len(s) and (s[i] == p[j] or p[j] == ".")
			print(f"    match {match} for {i}, {j}")
			if j+1 < len(p) and p[j+1] =="*": 				# if next p[j] is a *
				print("::::: Found *")
				cache[(i, j)] = dfs(i, j+2) or (match and dfs(i+1, j))
				
				print("####### Returned with cache #########")
				return cache[(i, j)]
			
			if match:
				print("[[[[[[[[[[[Same letter]]]]]]]]]]]]]")
				cache[(i, j )] =dfs(i+1, j+1)
				
				return cache[(i, j)]
			return False
		data = dfs(0, 0)
		print("\n\n",cache)
		return data
				
	def mergeKLists(self, lists):
		# 23
		def combineLists(l1, l2):
			dummy = ListNode()
			tail = dummy
			while l1 and l2:
				if l1.val > l2.val:
					tail.next = l2
					l2 = l2.next
				else:
					tail.next = l1
					l1 = l1.next
				tail = tail.next
			if l1:
				tail.next = l1
			if l2:
				tail.next = l2
			# showNodes(dummy)
			return dummy.next

		if not list or len(lists) == 0:
			return None
		while len(lists)>1:
			mergedList = []
			for i in range(0, len(lists), 2):
				l1 = lists[i]
				l2 = lists[i+1] if (i +1)  < len(lists) else None
				mergedList.append(combineLists(l1, l2))
			lists = mergedList
		# print(mergedList)
		return lists[0]
		
	def reverseKGroup(self, head, k):
		# 25
		def getKth(curr, k):
			while curr and k>0:
				curr = curr.next
				k-=1
			return curr
		# showNodes(head)
		dumy = ListNode(0,head)
		groupPrev = dumy

		while True:
			kth = getKth(groupPrev, k)
			if not kth:
				print("exit----------")
				break
			groupNxt = kth.next 
			prev, curr = kth.next, groupPrev.next
			print(f"kth= {kth}, grprv= {groupPrev}, grpnxt= {groupNxt}, c= {curr}, p= {prev}")
			while curr != groupNxt:
				tmp = curr.next 
				# print(f"{curr}, {prev}, temp = {tmp}")
				curr.next = prev
				prev = curr
				curr = tmp
				print(curr, "end", prev)
			tmp = groupPrev.next
			print(f"temp = {tmp}")
			groupPrev.next = kth
			groupPrev = tmp
		return showNodes(dumy).next

	def findSubstring(self, s, words):
		# 30
		print(words, s)
		wordLen = len(words[0])
		subStrLen = len(words) * len(words[0])
		res = []
		for left in range(wordLen): # 3
			d = collections.Counter(words)
			for right in range(left, len(s) + 1 - wordLen, wordLen):
				
				word = s[right: right + wordLen]
				d[word] -= 1
				while d[word] < 0:
					d[s[left: left + wordLen]] += 1
					left += wordLen
				if left + subStrLen == right + wordLen:
					res. append(left)			
		return res
		
	def longestValidParentheses(self,s):
		# 32
		print(s,'\n')
		i= 0 
		maxx = 0
		l,r = 0,0
		while i<len(s):
			if s[i] =="(":
				l+=1
			elif s[i] == ")":
				r+=1
			if r==l:
				maxx = max(maxx, l+r)
			elif l < r:
				l,r = 0,0
			i+=1
		l, r = 0,0
		i = len(s) -1
		while i >= 0:

			if s[i] =="(":
				l += 1
			elif s[i] == ")":
				r += 1
			if r == l:
				maxx = max(maxx, l+r)
			if l > r:
				l,r = 0,0
			i-=1
		return maxx

	def solveSudoku(self, board):
		# 37
		N=len(board)
		def printData():
			for i in range(N):
				if i % 3 == 0 and i != 0:
					print("-" * 21)  

				for j in range(N):
					if j % 3 == 0 and j != 0:
						print("|", end=" ") 

					print(board[i][j] if board[i][j] != "." else ".", end=" ")

				print()
		printData()

		rows = [set() for _ in range(9)]
		cols = [set() for _ in range(9)]
		box = [set() for _ in range(9)]
		empty = []

		for r in range(9):
			for c in range(9):
				val = board[r][c]
				if val == ".":
					empty.append((r, c))
				else:
					rows[r].add(val)
					cols[c].add(val)
					box[(r // 3) * 3 + c // 3].add(val)
		print()
		def backtrack(index):
			if index == len(empty):
				return True
			r, c = empty[index]
			b = (r//3)*3 + c//3

			for num in '123456789':
				if num in rows[r] or num in cols[c] or num in box[b]:
					continue
				board[r][c] = num
				rows[r].add(num)
				cols[c].add(num)
				box[b].add(num)

				if backtrack(index+1):
					return True
				board[r][c] = "."
				rows[r].remove(num)
				cols[c].remove(num)
				box[b].remove(num)
			return False


		backtrack(0)
		printData()
		return board
	
	def firstMissingPositive(self, A):
		# 41
		N = len(A)
		print(A, "lenght = ",N)
		for i in range(N):
			if A[i]<0:
				A[i] = 0
		print(A)
		print("Loop 2")
		for i in range(N):
			val = abs(A[i])
			print(f"trying with val {val}")
			if 1<=val <=N:
				print(f"Val = {val}, i = = {i}, A[prev] = {A[val -1]}")
				if A[val-1] > 0:
					A[val - 1] *= -1
				elif A[val -1] == 0:
					A[val -1 ] = -1 * (N+1)
					print(f"made lenght * -1 {A[val-1]}")
				print(f"new prev val = {A[val-1]} to {val -1}")
			print(f"Array now {A} \n\n")
		print("After second loop ",A)
		for i in range(1,N+1):
			if A[i - 1] >= 0:
				print(f"from loop, {i}, calc = {A[i-1]}")
				return i
		return N +1

	def trap(self, height):
		
		# 42
		if not height:
			return 0
		print(height)
		N = len(height)
		l, r = 0, N - 1
		maxL = height[l] 
		maxR = height[r]
		res = 0
		while l<r:
			maxL = max(maxL, height[l])
			maxR = max(maxR, height[r])
			# print_array_with_pointers(height,l,r,f"   l= {l}, r= {r}, ML= {maxL}, MR= {maxR}, re= {res}",True) 
			if height[l] < height[r]:
				res = res + (maxL - height[l]) if maxL - height[l]> 0 else res
				l+=1
			else:
				res = res + (maxR - height[r]) if maxR - height[r]> 0 else res
				
				r-=1 
		print(res)

	def isMatch2(self, s, p):
		# 44
		
		if p == "*":
			return True
		NS, NP = len(s), len(p)
		
		hasCache= [[False]*NP for _ in range(NS)]
		cach = [[None] * NP for _ in  range(NS)]
		def ismatch(index_s, index_p):
			
			if index_p == NP and index_s == NS:
				return True
			
			if index_p == NP:
				return False
			

			if  index_s == NS:
				if p[index_p] =="*":
					return ismatch(index_s, index_p+1) 
				return False
			
			if hasCache[index_s][index_p]:
				return cach[index_s][index_p]


			if p[index_p] == "*":
				cach[index_s][ index_p] = ismatch(index_s+1, index_p) or ismatch(index_s, index_p+1)
				hasCache[index_s][index_p] = True
				return cach[index_s] [index_p]
				
			if s[index_s] == p[index_p] or p[index_p] == "?":
				cach[index_s][ index_p] = ismatch(index_s+1, index_p+1)
				hasCache[index_s][index_p] = True
				return cach[index_s] [index_p]

			hasCache[index_s][index_p] = True
			cach[index_s][ index_p] = False
			return cach[index_s] [index_p]
				
		return ismatch(0,0)
			
	def solveNQueens(self, n):
		# 51
		print(n)
		board = [["." for _ in range(n)] for _ in range(n)]
		cols = set()
		posDia = set()
		negDia = set()
		
		res=[]
		print_matrix(board)
		def place(r):
			if r==n:
				copy = ["".join(row) for row in board ]
				res.append(copy)
				return
			
			for c in range(n):
				if c in cols or r+c in posDia or r-c in negDia:
					continue
				cols.add(c)
				posDia.add(r+c)
				negDia.add(r-c)
				board[r][c] = "Q"

				place(r+1)
				cols.remove(c)
				posDia.remove(r+c)
				negDia.remove(r-c)
				board[r][c] = "."
		place(0)
		
		return res
		
	def totalNQueens(self, n):
		# 52
		board = [[''] * n for _ in range(n)]

		cols = set()
		posDia = set()
		negDia = set()

		res = [0]
		def place(r):
			if r==n:
				res[0]+=1
			for c in range(n):
				if c in cols or r+c in posDia or r-c in negDia:
					continue
				board[r][c] = "Q"
				negDia.add(r-c)
				posDia.add(r+c)
				cols.add(c)
				place(r+1)
				board[r][c] = "."
				negDia.remove(r-c)
				posDia.remove(r+c)
				cols.remove(c)

		place(0)
		return res[0]




s = HardSolution()

test_arg1 = 4
test_arg2 = ['b',"*?*?"]
passes = test_arg1
leetcode_output( 51, s.totalNQueens ,passes ) #  // Output: 2
# print(out)








# leetcode_output( 4,s.findMedianSortedArrays, [1], [2]) #  // Output: 1.500
# leetcode_output( 10,s.isMatch, 'aa', "a*") #  // Output: True
# leetcode_output( 23,s.mergeKLists, [buildNodes([1,4,5]), buildNodes([1,3,4]), buildNodes([2,6])]) #  // Output: 1->1->2->3->4->4->5->6 
# leetcode_output( 25,s.reverseKGroup, buildNodes([1,2,3,4,5]), 2) #  // Output: 2-> 1-> 4-> 3-> 6-> 5-> |
# leetcode_output( 30, s.findSubstring, 'barfoofoobarthefoobarman', ["bar","foo","the"]) #  // Output:  [6,9,12]
# leetcode_output( 32, s.longestValidParentheses, ')()())') #  // Output:  4
# out = leetcode_output( 37, s.solveSudoku, [[".",".",".",".",".",".",".",".","."],[".","9",".",".","1",".",".","3","."],[".",".","6",".","2",".","7",".","."],[".",".",".","3",".","4",".",".","."],["2","1",".",".",".",".",".","9","8"],[".",".",".",".",".",".",".",".","."],[".",".","2","5",".","6","4",".","."],[".","8",".",".",".",".",".","1","."],[".",".",".",".",".",".",".",".","."]]) #  // Output:  [['7', '2', '1', '8', '5', '3', '9', '4', '6'], ['4', '9', '5', '6', '1', '7', '8', '3', '2'], ['8', '3', '6', '4', '2', '9', '7', '5', '1'], ['9', '6', '7', '3', '8', '4', '1', '2', '5'], ['2', '1', '4', '7', '6', '5', '3', '9', '8'], ['3', '5', '8', '2', '9', '1', '6', '7', '4'], ['1', '7', '2', '5', '3', '6', '4', '8', '9'], ['6', '8', '3', '9', '4', '2', '5', '1', '7'], ['5', '4', '9', '1', '7', '8', '2', '6', '3']]
# out = leetcode_output( 41, s.firstMissingPositive, [1,2,0] ) #  // Output:  3
# out = leetcode_output( 42, s.trap, [0,1,0,2,1,0,1,3,2,1,2,1]) #  // Output:  6
# out = leetcode_output( 44, s.isMatch2, "adceb", "*a*b" ) #  // Output:  True
# leetcode_output( 51, s.solveNQueens, 4 ) #  // Output: [[".Q..","...Q","Q...","..Q."]