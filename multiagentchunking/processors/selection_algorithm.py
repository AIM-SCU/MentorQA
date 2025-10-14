import random
from collections import defaultdict
import math
from typing import List, Dict

class SelectionAlgorithm:
    def segment_quality_score(self, items, score_key="score", seg_key="segment_id"):
        """
        Compute a 'quality' score per segment: quality = mean(score) - sd(score).
        Returns: dict {segment_id: quality}
        """
        by_seg = defaultdict(list)
        for x in items:
            by_seg[x[seg_key]].append(x[score_key])

        qualities = {}
        for seg, vals in by_seg.items():
            if not vals:
                qualities[seg] = 0.0
                continue
            mu = sum(vals) / len(vals)
            # compute standard deviation, fallback to 0 when len=1
            sd = (sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            qualities[seg] = mu - sd
        return qualities

    def compute_quotas(self, segments, K):
        """
        This function computes the quotas per segment based on their quality score.
        For segment i, the quota is computed by
        quota_i = floor ((score_i / total_score)* K )
        """
        total_score = sum(segments.values())
        
        raw = {s: (v / total_score) * K for s, v in segments.items()}
        # Floor first
        quotas = {s: math.floor(x) for s, x in raw.items()}

        # Distribute remainder by largest fractional part
        remainder = K - sum(quotas.values())
        if remainder > 0:
            order = sorted(segments.keys(), key=lambda s: (raw[s] - math.floor(raw[s])), 
                        reverse=True)
            for s in order:
                if remainder == 0:
                    break
                quotas[s] += 1
                remainder -= 1
        elif remainder < 0:
            # (rare) if flooring overshot due to numerical quirks
            order = sorted(segments.keys(), key=lambda s: (raw[s] - math.floor(raw[s])))
            for s in order:
                if remainder == 0:
                    break
                if quotas[s] > 0:
                    quotas[s] -= 1
                    remainder += 1

        return quotas

    def select_proportionally_distributed(self, items, K, skip_topics=None):
        """
        items: [{id, segment_id, score, text}]
        Returns ids of K items, roughly even across segments. 
        Skip the particular topics (introduction)
        """
        skip_topics = {t.lower() for t in (skip_topics or set())}

        # --- remove any items from skip topics ---
        # might update this part
        filtered_items = [
            x for x in items
            if x.get("segment_topic", "").lower() not in skip_topics
        ]
        # calculate the score of the segment
        selection_algorithm = SelectionAlgorithm()
        segment_score = selection_algorithm.segment_quality_score(
            filtered_items, score_key="score", seg_key="segment_id")
        ## DEBUGGING
        print(f"The segment score is {segment_score}")
        # quotas
        quotas = selection_algorithm.compute_quotas(segment_score, K)
        ## DEBUGGING
        print(f"The quota for each segment is: {quotas}")

        # pick top scores within each segment
        by_seg = defaultdict(list)
        for x in filtered_items: 
            by_seg[x["segment_id"]].append(x)
            
        selected, leftovers = [], []
        for s in sorted(by_seg.keys()):
            cand = sorted(by_seg[s], key=lambda x: (x["score"],x["id"]), reverse=True)
            take = min(quotas.get(s,0), len(cand))
            selected.extend(cand[:take])
            leftovers.extend(cand[take:])

        # fill remainder globally by score score
        if len(selected) < K:
            leftovers.sort(key=lambda x: (x["score"], x["id"]), reverse=True)
            selected.extend(leftovers[: K - len(selected)])

        return [x["id"] for x in selected[:K]]