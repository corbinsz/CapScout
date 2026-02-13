import { getScoreColor, getScoreBg, getScoreLabel, getScoreGradient } from '../../utils/scoreHelpers';
import { formatScore } from '../../utils/formatters';

export default function ScoreBadge({ score, variant = 'tag' }) {
  if (variant === 'bar') {
    return (
      <div className="score-bar" style={{ background: getScoreGradient(score) }}>
        <span className="score-bar-value">{formatScore(score)}</span>
        <div className="score-bar-info">
          <span className="score-bar-label">Investment Score</span>
          <span className="score-bar-desc">{getScoreLabel(score)}</span>
        </div>
      </div>
    );
  }

  return (
    <span
      className="score-tag"
      style={{ backgroundColor: getScoreBg(score), color: getScoreColor(score) }}
      title={getScoreLabel(score)}
    >
      {formatScore(score)}
    </span>
  );
}
