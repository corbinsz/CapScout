export default function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-message">
      <p>{typeof message === 'string' ? message : 'Something went wrong.'}</p>
      {onRetry && <button onClick={onRetry}>Retry</button>}
    </div>
  );
}
