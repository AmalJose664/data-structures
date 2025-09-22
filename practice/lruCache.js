/**
 * 
 * Demo of lru cache for js hard questions
 */


class LRUCache {

	constructor(capacity) {
		this.capacity = Number(capacity)
		this.head = null
		this.map = new Map()
		this.tail = null
		this.length = 0
	}
	#removeNode(node) {
		console.log("Removing node :", node.key)
		if(!node) return
		if (node.prev) {
			node.prev.next = node.next
		}
		if(node.next){
			node.next.prev = node.prev
		}
		if(node === this.head){
			this.head = node.next
		}
		if(node === this.tail){
			this.tail = node.prev
		}
	}
	get(key) {
		if(!this.map.has(key)){
			return null
		}
		const node = this.map.get(key)
		this.#removeNode(node)
		node.prev = null
		node.next = this.head
		if(this.head !== null){
			this.head.prev = node
		}
		this.head = node
		return node
	}
	put(key, value) {

		if(this.length == this.capacity){
			if(!this.map.has(key)){
				this.#removeNode(this.tail)
				this.length -=1
			}
		}
		if (this.map.has(key)) { // if key exists already
			this.#removeNode(this.map.get(key))
		}
		const node = {
			value, key,
			prev: null,
			next: this.head
		}
		this.map.set(key, node)
		if(this.head !== null){
			this.head.prev = node
		}
		this.head = node

		if(this.tail === null){
			this.tail = node
		}

		this.length += 1

	}
	debug(){
		let  curr = this.head
		const arr = []
		while(curr !== null){
			arr.push(curr)
			curr = curr.next
		}

		return arr.reduce((acc, cur) => acc.concat(`-->[ [${cur.key}]: [${cur.value}] ]-->`) ,"")
	}
}


// const cache = new LRUCache(2)

// cache.put(1, 10)
// cache.put(2, 20)
// // console.log(cache.get(1))
// cache.put(3, 30)

//console.log(cache.debug())

class CustomLRU{
	constructor(capacity){
		this.capacity = Number(capacity)
		this.head = null
		this.tail = null
		this.length = 0
		this.map = new Map()
	}

	removeNode(node){
		if(!node) return null

		if(node.next === null){
			if(node === this.head){
				this.head = null
				this.tail = null
				this.length -=1
				return
			}
			this.tail = node.prev
			node.prev.next = null
			this.length -=1
			return
		}
		if(node.prev === null){
			node.next.prev = null
			this.head = node.next || null
			this.length -=1
			return
		}
		node.prev.next = node.next
		node.next.prev = node.prev
		this.length -=1

	}

	put(key, value){
		if(this.map.has(key)){
			this.removeNode(this.map.get(key))
		}
		if(this.capacity == this.length){
			this.removeNode(this.tail)
		}
		const node = {
			key, value,
			prev: null,
			next: this.head
		}
		if(this.head !== null){
			this.head.prev = node
		}
		this.map.set(key, node)
		this.head = node
		if(this.tail === null){
			this.tail = node
		}
		
		this.length +=1

	}
	print(){
		let curr = this.head
		const arr = []
		while(curr){
			const {key, value} = curr
			arr.push({key, value})
			curr = curr.next
		}	
		console.log(arr, this.length)
	}

}
const c = new CustomLRU(2)
console.log(c)
c.put(1, 5)
c.put(2, 9)
// c.put(1, 25)
// c.put(2, 16)

c.put(4, 90)
// c.removeNode(c.map.get(4))
// c.removeNode(c.map.get(3))
// c.removeNode(c.map.get(4))
// c.removeNode(c.map.get(1))
c.print()
