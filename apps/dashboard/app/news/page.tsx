"use client";

/**
 * News and research.
 *
 * Both timestamps are shown. The gap between publication and ingestion is what a
 * backtest must respect: several free government feeds back-date items, so
 * filtering on the published time would hand a simulator hours of hindsight.
 */

import { api, timeAgo, type NewsRow } from "@/lib/api";
import { ErrorBox, Empty, Pill, usePoll } from "@/components/ui";

export default function NewsPage() {
  const { data, error } = usePoll<{ articles: NewsRow[] }>(() => api.news(), 10000);
  const articles = data?.articles ?? [];

  return (
    <>
      <h1>News &amp; research</h1>
      <div className="sub">
        Ingested articles, extracted entities, and the markets they were matched to.
      </div>
      <ErrorBox error={error} />

      {articles.length === 0 ? (
        <Empty>
          No articles ingested yet. The news pipeline polls free RSS and public government
          feeds; it never requires a paid API.
        </Empty>
      ) : (
        <div className="grid g2">
          {articles.map((a) => (
            <div key={a.article_id} className="card">
              <div className="row small muted" style={{ marginBottom: 6 }}>
                <Pill>{a.source}</Pill>
                <span>published {timeAgo(a.published_ts)}</span>
                <span>ingested {timeAgo(a.ingested_ts)}</span>
                {a.source_reliability !== null ? (
                  <span>reliability {a.source_reliability.toFixed(2)}</span>
                ) : null}
              </div>
              <div style={{ fontWeight: 600, marginBottom: 5 }}>
                {a.url ? (
                  <a href={a.url} target="_blank" rel="noreferrer">
                    {a.title}
                  </a>
                ) : (
                  a.title
                )}
              </div>
              {a.summary ? <div className="small muted">{a.summary.slice(0, 260)}</div> : null}
              {a.markets.length > 0 ? (
                <div className="row small" style={{ marginTop: 8 }}>
                  {a.markets.slice(0, 6).map((m) => (
                    <Pill key={m.ticker} tone={m.relevance > 0.6 ? "good" : "neutral"}>
                      {m.ticker} {m.relevance.toFixed(2)}
                    </Pill>
                  ))}
                </div>
              ) : (
                <div className="note">Matched to no market.</div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
}
