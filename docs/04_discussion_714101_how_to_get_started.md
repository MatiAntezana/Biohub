# Discussion: How to get started + Competition's Official Discord
- **Topic URL:** https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/714101
- **Author:** María Cruz (@macruzbar)
- **Date:** 2026-06-25T16:44:43.592999100Z
- **Votes:** 24
- **Total Messages:** 7

---

## Main Post

### Information for newbies
**New to machine learning and data science?** No question is too basic or too simple. Feel free to start your own thread, or use this thread as a place to post any first-timer clarifying questions for the Kaggle community to help you with!

**New to Kaggle?** Take a look at a few videos to learn a bit more about [site etiquette](https://www.youtube.com/watch?v=aIus8si_Et0), [Kaggle lingo](https://www.youtube.com/watch?v=sEJHyuWKd-s), and [how to enter a competition using Kaggle Notebooks](https://www.youtube.com/watch?&v=GJBOMWpLpTQ). Publish and share your [models on Kaggle Models](https://www.kaggle.com/docs/models#publishing-a-model)!

**Looking for a team?** Express your interest in joining a team through our [Team Up](https://www.kaggle.com/discussions/product-feedback/341195) feature.

**Remember**: Kaggle is for everyone. Whether you're teaming up or sharing tips in the competition forum, we expect everyone to follow our Kaggle community guidelines.

### Competition's Official Discord
In addition to this competition forum, you can continue the discussion in our official Kaggle Discord Server here:
# [discord.gg/kaggle](http://discord.gg/kaggle)
 
The Discord is a great place to ask getting started questions, chat about the nuances of this competition, and connect with potential team mates. Learn more about Discord at our [announcement here](https://www.kaggle.com/discussions/general/429933). Here are a few things to keep in mind though:

**1. Discord Competition Channels are 'Public' - Don't Share Private Information**

Discord channels for specific competitions are considered 'public' spaces where you are allowed to talk about competition details. Please remember that private sharing of competition code or data outside of your team is, as always, not permitted. Code sharing must always be done publicly through the Kaggle forums/notebooks.

**2. Discord Competition Channels are Not Monitored by Staff - Keep Important Information on the Kaggle Forums**

Kaggle Staff and Hosts running competitions will not monitor Discord or be available to answer questions in Discord. This is intended to be a more casual space to discuss competitions and help each other. Please keep important questions, insights, writeups, and other valuable conversation on the Kaggle forums. 

Happy modeling!

---

## Comments & Community Discussion

### Comment 1 by Cho Royou (@) - 2026-07-17T15:45:54.417Z

Hi organizers,

This public notebook appears to exploit a weakness in how the official scoring script parses submitted submission.csv files, rather than reflecting genuine cell tracking accuracy:

https://www.kaggle.com/code/outwrest/metric-hack-minimal-baseline-tta-2gpu

The detection/tracking pipeline itself (TemporalUNet3D + node transformer + ILP solver, TTA, multi-GPU inference) is a legitimate baseline. The problem is a post-processing cell (augment_dataset) that runs after predictions are generated and modifies only the final submitted CSV — it does not affect the notebook's own local metric check, which is a strong signal it's specifically targeting the official scorer rather than improving predictions.

For each sample, it:

Splits the predicted lineage forest into connected components, sorts by size, and keeps up to MAX_COMPONENTS = 1400 of the largest trees.

Adds one synthetic "hub" node at sentinel coordinates (t=-1000, z=y=x=-10000 — clearly outside any real acquisition volume/time range) connected to the roots of those trees.

Appends a chain of 5 synthetic cell-division events (divider → child, continuation), also at sentinel coordinates (t≈-999..., z=y=x=-10000/-10001).

None of this corresponds to real detections — the coordinates are deliberately outside any physically valid range, so it can't be genuine matched signal under normal ground-truth matching. It looks designed to exploit something in how the scorer counts connected components / division events / normalizes its Jaccard-style metric.

This is a public, forkable notebook, so it may already be in use by other teams — I'd flag this as urgent. I forked/submitted a copy of it myself (submission ID 54786048, 2026-07-17) before fully realizing what the post-processing step did, and would like that submission disregarded/excluded from scoring.

Happy to provide more details if useful.

### Comment 2 by Evdilos_Ikaria (@) - 2026-07-03T09:26:08.437Z

Is there any available code for the applied combined tracking metric?
 Thanks

### Comment 3 by axorina (@) - 2026-08-22T07:38:14.340Z

hi everyone im a 12th grader  and im looking for a beginner friendly team for biohub cell track if annyone is interested let me know!

### Comment 4 by Asogwa Samson (@) - 2026-07-31T16:53:46.037Z

Pardon me,
How long will it take for ones file to be processed?
