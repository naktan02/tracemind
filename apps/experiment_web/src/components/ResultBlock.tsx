export function ResultBlock(props: { title: string; lines: string[] }) {
  return (
    <section className="result-block">
      <h4>{props.title}</h4>
      {props.lines.length > 0 ? (
        <pre>{props.lines.join("\n")}</pre>
      ) : (
        <p className="hint-text">No values</p>
      )}
    </section>
  );
}
