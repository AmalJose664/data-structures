/**
 *  hey
 */

class EventEmitter{
	constructor(){
		// [event]: subscribers[]
		this.__event_listeners = {}
	}
	on(event, listener){
		// register for a event
		if(!this.__event_listeners[event]){
			this.__event_listeners[event] = []
		}
		this.__event_listeners[event].push(listener)
		return true
	}

	emit(event, ...args){
		if(!this.__event_listeners[event]){
			return false
		}
		const listeners = this.__event_listeners[event]
		listeners.forEach(listener => listener(...args))

	}
	off(event, listener){
		if(!this.__event_listeners[event]){
			return
		}
		this.__event_listeners[event] = this.__event_listeners[event].filter(l => l!== listener)
	}
	once(event, listener){
		const wrapperFn = (...args)=>{
			this.off(event, wrapperFn)
			listener(...args)
		}
		this.on(event, wrapperFn)
		return true
	}
}

const e = new EventEmitter()


const whatsapp = (username)=>{
	console.log("sending whatsapp to  ",username)
}
const signup = (username)=>{
	console.log("user has signed up ",username)
}
const email = (username)=>{
	console.log("sending email to  ",username)
}
const logout = (username)=>{
	console.log("user logout  ",username)
}

e.on("user:signup", signup)
e.once("user:signup", email)
e.once("user:signup", whatsapp)

e.on("user:logout", logout)

e.emit("user:signup", "@piyushgarh")
console.log("------------------------")
e.emit("user:signup", "@tony")
console.log("------------------------")


e.off("user:signup", whatsapp)

e.emit("user:signup", "@friday")
console.log("------------------------")
e.emit("user:logout", "@piyushgarh")
console.log("------------------------")