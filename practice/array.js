//--------------------------------------------------------------------------------------------------------
class MyArray {
    constructor() {
        this.length = 0
        this.data = {}
    }
    show() {
        console.log(this.data, "length = ", this.length)
    }
    push(value) {
        this.data[this.length] = value
        this.length++
    }
    get(index) {
        if (this.length > index) {
            return this.data[index]
        } else {
            return "Out of bounds"
        }
    }
    pop() {
        delete this.data[this.length - 1]
        this.length--
        return this.length + 1
    }
    shift() {
        const returnItem = this.data[0]
        for (let i = 0; i < this.length; i++) {
            this.data[i] = this.data[i + 1]
        }
        this.pop()
        return returnItem
    }
    deleteByIndex(pos) {
        if (this.length > pos) {
            const returnItem = this.data[pos]
            for (let i = pos; i < this.length; i++) {
                this.data[i] = this.data[i + 1]
            }
            this.pop()
            return returnItem
        } else {
            return "Out of bounds"
        }
    }
}
function reverseString(str) {
    let newStr = ""
    for (i = str.length - 1; i >= 0; i--) {
        newStr = newStr + str[i]
    }
    return newStr
}
function reverseNumber(num) {
    let rev_num = 0
    while (num > 0) {
        rev_num = rev_num * 10 + (num % 10)
        num = Number((num / 10).toFixed())
    }
    return rev_num
}
const palindromeNum = (num) => reverseNumber(num) === num
function palindromeStr(str) {
    return reverseString(str) === str
}

const capitalize = (str) => str.slice(0, 1).toUpperCase() + str.slice(1).toLowerCase()
const newArr = new MyArray()

// for (let i = 0; i <= 10; i++) {
//     newArr.push("Item => " + i)
// }
function fizzBuzz(n) {
    for (let i = 0; i <= n; i++) {
        if (i % 3 == 0) console.log("Fizz ==" + i)
        if (i % 5 == 0) console.log("Buzz==" + i)
        if (i % 5 == 0 && i % 3 == 0) console.log("FizzBuzz==" + i)
    }
    return "\nFinished"
}
//console.log("get ", newArr.get(4))
//console.log(newArr.pop())
// console.log(newArr.shift(0))
// console.log(newArr.deleteByIndex(6))
// newArr.show()

// console.log(reverseString("Heelos people"))
// console.log("Palindrome or not = ", palindromeStr("tenet"))
// console.log(reverseNumber(1002))
// console.log("Reverse or not ==", palindromeNum(2002))
// console.log(capitalize("hey peopEple"))
// console.log(fizzBuzz(20))
// challenge
// max profit

const maxProfit = (prices) => {
    let minPrice = prices[0]
    let maxProfit = 0

    for (let i = 1; i < prices.length; i++) {
        const currentPrice = prices[i]
        // minPrice = minPrice < currentPrice ? minPrice : currentPrice
        minPrice = Math.min(minPrice, currentPrice)

        const potentialPrice = currentPrice - minPrice
        maxProfit = Math.max(maxProfit, potentialPrice)

        console.log("Cuurent=> ", currentPrice, " minP=> ", minPrice, " MaxP=> ", maxProfit)
    }
    return maxProfit
}
const prices = [7, 1, 5, 3, 6, 4] //[5, 8, 9, 2, 1, 0, 3, 4]

// console.log("maxPorfit = ", maxProfit(prices))

const arrayChunk = (arr, chunkSize) => {
    const newArr = []
    let insertIndex = 0
    let index = 0
    // while (index < arr.length) {
    //     let chunk = []

    //     for (i = 0; i < chunkSize && index < arr.length; i++) {
    //         chunk[i] = arr[index]
    //         index++
    //     }

    //     newArr[insertIndex] = chunk
    //     insertIndex += 1
    // }
    while (index < arr.length) {
        const chunk = arr.slice(index, index + chunkSize)
        index += chunkSize
        newArr[insertIndex] = chunk
        insertIndex++
    }

    return newArr
}

// console.log("Result = ", arrayChunk([1, 2, 3, 4, 5, 6, 7, 8], 3))

const twoSums = (arr, target) => {
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length; j++) {
            if (arr[i] + arr[j] === target) {
                return [i, j]
            }
        }
    }
}
console.log(twoSums([2, 7, 11, 15], 9))
console.log(twoSums([1, 3, 7, 9, 2], 11))
