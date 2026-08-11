# The buying methodology

The purchasing problem is narrower than "detect duplicates", and the commands
map onto it directly. A lab is offered 10,000 hours. It has bought from other
vendors before, some of it through aggregators who may be reselling each other's
inventory, and it has consumed open datasets. The seller will share one or two
hours as a sample and nothing more before payment - and each seller picks a
different slice, so comparing samples proves nothing about the rest.

**1. Keep a fingerprint record of everything you have seen.** Not only what you
bought: what you declined, and the public datasets you have consumed. A declined
offer's manifest is worth keeping because it may come back wearing a
different aggregator's name.

```
overlap index /data/corpus                    # footage you hold
overlap import declined-2026q2.ovlm            # an offer you turned down
overlap import egocentric-100k.ovlm            # a public dataset, fingerprints only
```

**2. Ask for a manifest covering all of it.** Not the sample - the whole offer.
It carries fingerprints, not frames, so a seller can produce it without giving
anything away. Ask for it at full density (`--stride 1`): a manifest strided to
1 fps was sized for emailing, and measured against re-cut footage it recovers
40% of what a 4 fps manifest recovers.

**3. Check the manifest describes the footage you were shown.** This is what the
sample is actually good for. It is the only footage you hold pixels for, so it
can be fingerprinted and looked up in the manifest: if the sample came from the
offered data, nearly all of it must appear.

```
overlap audit-sample q3-offer.ovlm --sample /data/incoming/sample
```

**4. Ask how much you already own.**

```
overlap compare q3-offer.ovlm --html report.html
```

**5. Ask whether two sellers are offering the same data** - even when you own
neither yet, which is the case that costs labs the most through aggregators:

```
overlap compare aggregator-a.ovlm --against aggregator-b.ovlm
```

**6. Buy, then check the delivery against the quote, and index it** so the next
offer is screened against it too:

```
overlap verify q3-offer.ovlm --data /data/incoming/delivery
overlap index /data/incoming/delivery
```

What each step can and cannot establish is stated in the report it produces
rather than left to inference. In particular: steps 3 and 5 work on fingerprints
alone, so neither side has pixels and cropped copies cannot be detected there - only step 4, against footage you hold, has the crop geometries for that.

Prefer a UI? `overlap ui` serves a local web interface (works over SSH
port-forwarding, since your data usually lives on a server):

```
ssh -L 8377:127.0.0.1:8377 user@dataserver   # then open the printed URL
```

