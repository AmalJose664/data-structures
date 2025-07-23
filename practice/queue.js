class Node {
    constructor(value) {
        this.value = value
        this.next = null
    }
}

class Queue {
    constructor(value) {
        const nNode = new Node(value)
        this.first = nNode
        this.last = nNode
        this.length = 1
    }
    enqueue(value) {
        let nNode = new Node(value)
        if (this.length === 0) {
            this.first = nNode
            this.last = nNode
        }
        this.last.next = nNode
        this.last = nNode
        this.length++
    }
    dequeue() {
        if (this.length === 0) {
            return undefined
        }
        let temp = this.first
        if (this.length === 1) {
            this.first = null
            this.last = null
        }
        this.first = this.first.next
        temp.next = null
        this.length--

        return temp
    }
}
let myQ = new Queue(8)
myQ.enqueue(90)
console.log(myQ.dequeue())
console.log(myQ)
