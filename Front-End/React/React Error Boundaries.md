Error boundaries are special React components which catch JavaScript errors in their child component tree, and display fallback UI without crashing the entire app/website.

Without error boundaries, if a component throws an error during a render, lifecycle method, or a constructor, React will unmount the entire component tree below that point.

By using error boundaries we can isolate the error to just the part of the UI containing the error.

> [!CAUTION]
> Error Boundaries will not catch errors thrown inside event handlers, SSR, or in async code (`setTimeout`, promises, etc.).

## Example:

> [!Important]
> This example was built using [Vite](https://vite.dev/) with the [`react-error-boundary`](https://www.npmjs.com/package/react-error-boundary) npm package.

```tsx
"use client";
import { useState } from "react";
import "./App.css";
import { ErrorBoundary } from "react-error-boundary";

// Component that will throw an error during rendering
const BuggyCounter = ({ count }: { count: number }) => {
  if (count === 5) {
    throw new Error("This is a simulated error");
  }
  return <div>Current count: {count}</div>;
};

  

function App() {
  const [count, setCount] = useState(0);
 
  const handleClick = () => {
    setCount((count) => count + 1);
  };

  return (
    <>
      <h1>React Error Boundaries</h1>
      <div className="card">
        <ErrorBoundary
          fallbackRender={({ resetErrorBoundary }) => (
            <div className="error-boundary">
              <h3>
                A simulated error has occurred due to the counter hitting 5.
              </h3>
              <h3>AN ERROR HAS OCCURRED: Fallback UI</h3>
              <button onClick={resetErrorBoundary}>Reset</button>
            </div>
          )}
          onReset={() => {
            setCount(0);
          }}
        >
          <button onClick={handleClick}>
          count is {count}
          </button>
          
          <BuggyCounter count={count} />
        </ErrorBoundary>
      </div>
    </>
  );
}

export default App;
```

