## What is Node JS and why do we use it?

NodeJS is a JS runtime built on Googles V8 JS engine. Typically JS is ran in the browser, but what if we want to run JS code outside of the browser? This is where NodeJS comes in handy. Googles V8 engine is used to run code outside of the browser, whilst NodeJS acts as a runtime.

NodeJS provides functionality for developers to do things such as accessing file systems and adding networking capabilities, allowing JS to be used on the server side of an application.

NodeJS is single-threaded, which means it's great for building fast and scalable applications, which is useful for data intensive apps (Such as using databases). It should not be used for applications which require heavy CPU intensive server-side processing. For intensive apps, Ruby on rails, PHP, or [[Python]] would be a much better choice.

## How Node JS works

Node JS has two main dependencies, the `V8 engine`, and `libuv`. Libuv is an open source library which manages asynchronous input/output actions as well as providing node access to the underlying OS, the file system, and networking. 

Libuv also implements two of the most important features of Node JS, the event loop, and the thread pool. Libuv is entirely written in C++ and the V8 engine is written in JS and C++.

Node JS also has 4 smaller dependencies, `http-parser`, `c-ares`, `OpenSSL`, and `zlib`.
