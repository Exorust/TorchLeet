export default function SourcesSection() {
  return (
    <section id="sources" className="py-12">
      <div className="rounded-2xl border border-gray-200 bg-white/60 backdrop-blur-lg shadow-sm p-7">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          How company tags are sourced
        </h2>
        <p className="text-sm text-gray-600 mb-4 max-w-3xl">
          Every problem carries company tags with an explicit confidence level,
          so you always know how much to trust a tag:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-3xl">
          <div className="rounded-xl border border-gray-200 bg-white/70 p-4">
            <span className="bg-lavender-100 text-lavender-600 text-xs rounded-full px-2 py-0.5 font-medium">
              reported
            </span>
            <p className="text-sm text-gray-600 mt-2">
              A candidate reported being asked this problem in an actual
              interview loop at that company.
            </p>
          </div>
          <div className="rounded-xl border border-gray-200 bg-white/70 p-4">
            <span className="bg-lavender-100 text-lavender-600 text-xs rounded-full px-2 py-0.5 font-medium">
              inferred
            </span>
            <p className="text-sm text-gray-600 mt-2">
              We matched the problem to the company from public job posts and
              the topics their teams work on — a strong signal, but not a
              first-hand report.
            </p>
          </div>
        </div>
        <p className="text-xs text-foreground/50 mt-4">
          Full per-problem attributions live in{" "}
          <a
            href="https://github.com/Exorust/TorchLeet/blob/main/SOURCES.md"
            target="_blank"
            rel="noopener noreferrer"
            className="text-lavender-600 hover:underline"
          >
            SOURCES.md
          </a>{" "}
          on GitHub.
        </p>
      </div>
    </section>
  );
}
