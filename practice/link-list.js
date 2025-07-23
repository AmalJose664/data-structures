class Node {
    constructor(data) {
        this.head = data
        this.next = null
    }
}

class LinkedList {
    constructor(value) {
        this.head = new Node(value)
        this.tail = this.head
        this.length = 1
    }
    push(value) {
        let newNode = new Node(value)

        if (!this.head) {
            this.head = newNode
            this.tail = newNode
        } else {
            this.tail.next = newNode
            this.tail = newNode
        }

        this.length++
    }
    pop() {
        if (!this.head) {
            return null
        }
        let temp = this.head
        let prev = this.head
        while (temp.next) {
            prev = temp
            temp = prev.next
        }
        this.tail = prev
        this.tail.next = null
        this.length--
        if (this.length === 0) {
            this.head = null
            this.tail = null
        }
        return temp
    }
    unshift(value) {
        const newNode = new Node(value)
        if (!this.head) {
            this.head = newNode
            this.tail = newNode
        }
        newNode.next = this.head
        this.head = newNode

        this.length++
        return this
    }
    reverse() {
        let temp = this.head
        this.head = this.tail
        this.tail = temp

        let next = temp
        let prev = null

        for (let i = 0; i < this.length; i++) {
            next = temp.next
            temp.next = prev
            prev = temp
            temp = next
        }
    }
    shift() {
        if (!this.head) {
            return undefined
        }
        let temp = this.head

        this.head = this.head.next
        temp.next = null
        this.length--

        if (this.length == 0) {
            this.tail == null
        }
        return temp
    }
    getFirst() {
        return this.head
    }
    getLast() {
        if (!this.head) {
            return null
        }
        let temp = this.head
        while (temp) {
            if (!temp.next) {
                return temp
            } else {
                temp = temp.next
            }
        }
    }
    get(index) {
        let count = 0
        let temp = this.head
        while (temp) {
            if (count === index) {
                return temp
            }
            count++
            temp = temp.next
        }
        return null
    }
    set(index, value) {
        let temp = this.get(index)
        if (temp) {
            temp.head = value
            return true
        }
        return false
    }
    insert(index, value) {
        if (index === 0) {
            return this.unshift(value)
        }
        if (index === this.length) {
            return this.push(value)
        }

        const newNode = new Node(value)
        const temp = this.get(index - 1)
        newNode.next = temp.next
        temp.next = newNode

        this.length++
        return true
    }
    display() {
        console.log("Start\n")
        let temp = this.head
        while (temp) {
            console.log(temp.head + "=>")
            temp = temp.next
        }
        console.log("\nEnd")
    }
    size() {
        let counter = 0
        let temp = this.head
        while (temp) {
            counter++
            temp = temp.next
        }
        return counter
    }
    clear() {
        this.head = null
        this.tail = null
    }
}

const myLinklist = new LinkedList(1)
myLinklist.push(10)
// console.log(myLinklist)
// myLinklist.unshift(89)
// console.log(myLinklist)
// console.log(myLinklist.shift())
// console.log(myLinklist.getFirst())
myLinklist.push(20)
myLinklist.push(40)
myLinklist.push(60)
myLinklist.push(80)
// console.log("Last == ", myLinklist.getLast())
// console.log("Get == ", myLinklist.get(1))
// console.log("Set == ", myLinklist.set(1, 980))

//console.log(myLinklist.insert(1, 94))
console.log("Size == ", myLinklist.size())
//myLinklist.clear()
//console.log("Size == ", myLinklist.size())

myLinklist.display()
myLinklist.reverse()
console.log(myLinklist)
myLinklist.display()
// console.log(myLinklist)
