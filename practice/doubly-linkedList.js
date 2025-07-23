class Node {
    constructor(value) {
        this.value = value
        this.next = null
        this.prev = null
    }
}
class DoublyLinkedList {
    constructor(value) {
        this.head = new Node(value)
        this.tail = this.head
        this.length = 1
    }
    push(value) {
        const newNode = new Node(value)
        if (!this.head) {
            this.head = newNode
            this.tail = newNode
        }
        this.tail.next = newNode
        newNode.prev = this.tail
        this.tail = newNode
        this.length++
        return this
    }
    pop() {
        if (this.length === 0) {
            return undefined
        }
        let temp = this.tail
        if (this.length === 1) {
            this.head = null
            this.tail = null
        }
        this.tail = this.tail.prev
        this.tail.next = null
        temp.prev = null

        this.length--
    }
    unshift(value) {
        let newNode = new Node(value)

        if (this.length === 0) {
            this.head = newNode
            this.tail = newNode
            this.length++
            return ""
        }

        newNode.next = this.head
        this.head.prev = newNode
        this.head = newNode
        this.length++
        return this
    }
    shift() {
        if (this.length === 0) {
            return undefined
        }
        let temp = this.head
        if (this.length === 1) {
            this.head = null
            this.tail = null
        }
        this.head = this.head.next
        this.head.prev = null
        temp.next = null
        this.length--
    }
    reverse() {
        let temp = this.tail
        this.head = this.tail
        while (this.head) {
            this.head.next = this.head.prev
        }
    }
}

const myDoubleLinklist = new DoublyLinkedList(11)

myDoubleLinklist.push(67)
myDoubleLinklist.push(90)
myDoubleLinklist.push(71)
myDoubleLinklist.pop()
myDoubleLinklist.unshift(25)
myDoubleLinklist.unshift(9)
myDoubleLinklist.shift()

console.log(myDoubleLinklist)
