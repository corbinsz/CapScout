export default function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-message">
      <p>{message || 'Something went wrong.'}</p>
      {onRetry && <button onClick={onRetry}>Retry</button>}
    </div>
  );
}
