class Node {
	constructor(value) {
		this.value = value
		this.left = null
		this.right = null
	}
}
class BST {
	constructor() {
		this.root = null
	}
	insert(value) {
		const nNode = new Node(value)

		if (this.root === null) {
			this.root = nNode
			return this
		}
		let temp = this.root
		while (true) {
			if (nNode.value === temp.value) {
				return undefined
			} else if (nNode.value < temp.value) {
				if (temp.left === null) {
					temp.left = nNode
					return this
				} else {
					temp = temp.left
				}
			} else {
				if (temp.right === null) {
					temp.right = nNode
					return this
				} else {
					temp = temp.right
				}
			}
		}
	}
	includes(value) {
		let temp = this.root
		while (temp) {
			if (temp.value === value) {
				return true
			} else if (temp.value < value) {
				temp = temp.right
			} else {
				temp = temp.left
			}
		}
		return false
	}
	bfs() {
		let current = this.root
		let queue = []
		let data = []

		queue.push(current)
		while (queue.length) {
			current = queue.shift()
			data.push(current.value)

			if (current.left) {
				console.log("inserting data=>", current.value)
				queue.push(current.left)
			}
			if (current.right) {
				console.log("inserting data=>", current.value)
				queue.push(current.right)
			}
		}
		return data
	}
	dfsPreOrder(node = this.root, data = []) {
		if (node === null) {
			return data
		}

		data.push(node.value)

		if (node.left) this.dfsPreorder(node.left, data)
		if (node.right) this.dfsPreorder(node.right, data)
		return data
	}
	dfsPostOrder(node = this.root, data = []) {
		if (node === null) {
			return data
		}

		if (node.left) this.dfsPostOrder(node.left, data)
		if (node.right) this.dfsPostOrder(node.right, data)
		data.push(node.value)
		return data
	}
	dfsInOrder(node = this.root, data = []) {
		if (node === null) {
			return data
		}

		if (node.left) this.dfsInOrder(node.left, data)
		data.push(node.value)
		if (node.right) this.dfsInOrder(node.right, data)
		return data
	}
}

const tree = new BST()

tree.insert(5)
// console.log(tree.includes(12))
console.log(tree.dfsInOrder())
console.log(tree)
