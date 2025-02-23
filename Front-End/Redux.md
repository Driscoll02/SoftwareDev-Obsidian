Store = Where the information/data is kept
Reducers = Reducers are used to update the store (adding data, removing data, updating data) and is usually just an object.
Slice = Used to track the initial state of a store, as well as all of its reducers.

useSelector = A hook which is used to talk to the data store. Usually used to ask for data.
useDispatch = A hook used to call a reducer to add or update data in the store.
