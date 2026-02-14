/**
 * Task Scheduler for executing a amount of tasks at a time.
 * Excess tasks are  stored to a queue and executed upon finshing of running tasks
 * 
 */
class TaskScheduler{
	constructor(concurrency){
		this.concurrency = Number(concurrency)
		this.runningTasks = 0
		this.__waitingQueue = []
	}

	getNextTask(){
		if(this.runningTasks  < this.concurrency && this.__waitingQueue.length > 0){
			const nextTask = this.__waitingQueue.shift()
			nextTask()
		}
	}
	addTask(task){
		return new Promise((resolve, reject)=> {
			async function __taskRunner(){
				this.runningTasks += 1
				try {
					const result = await task()
					console.log(result, " result ")
					resolve(result)
				} catch (err) {
					console.log(err, "task faiiled ")
					reject(err)
				}finally{
					this.runningTasks -=1 
					// see if there any tasks in queue
					// if so run that
					this.getNextTask()
				}
			}
			if(this.runningTasks < this.concurrency){
				__taskRunner.call(this)
			}else{
				this.__waitingQueue.push(__taskRunner.bind(this))
				console.log(this.__waitingQueue, this.__waitingQueue.length)
			}
		})
	}
}


const s = new TaskScheduler(30)

// s.addTask(
// 	()=> new Promise(res => setTimeout(() => {
// 		res("Task 1")
// 	}, 3*1000))
// )
// s.addTask(
// 	()=> new Promise(res => setTimeout(() => {
// 		res("Task 2")
// 	}, 3*1000))
// )
// s.addTask(
// 	()=> new Promise(res => setTimeout(() => {
// 		res("Task 3")
// 	}, 3*1000))
// )
// s.addTask(
// 	()=> new Promise(res => setTimeout(() => {
// 		res("Task 4")
// 	}, 3*1000))
// )

function saveToDb(message) {
	return new Promise((res, rej)=> setTimeout(()=>{
		console.log(`${message} saved to db!`)
		res("----Done "+ message + " ----")
	}, 0 ))
}

function chat(){
	const msg = Array(100).fill(null)

	msg.forEach((_,i)=>{
		const message = "Message: "+i
		s.addTask(()=>saveToDb(message))
	})
}
chat()