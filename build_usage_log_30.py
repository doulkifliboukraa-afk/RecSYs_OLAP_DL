"""
build_usage_log_30.py
=====================
Extends the original 15-persona simulation to 30 personas,
keeping the same StackExchange polystore schema (15 datasets, 63 attrs).
The 15 new personas (F1-F5, G1-G5, H1-H5, I1-I3, J1-J2) cover
analytical angles not represented in the original set:
badge analytics, cross-store comment analysis, temporal vote patterns,
link-graph exploration, and tag-centric cube design.
"""
import random
from build_usage_log import (
    DATASETS, ATTRS, PERSONAS as PERSONAS_15,
    COLD, attrs_of, find, events as events_15
)

# ── 15 new personas ────────────────────────────────────────────────────────
NEW_PERSONAS = {
 # Cluster F: Neo4j-centric analytics
 "F1": dict(cluster="F", fact="SE_neo4j.Posts",
    alt_facts=["SE_mysql.postes"],
    dims=["SE_neo4j.Users","SE_neo4j.Tags"], dim_alt=["SE_neo4j.Comments"],
    measures=[find("SE_neo4j.Posts","posttypeid"), find("SE_neo4j.Posts","owneruserid")],
    params={"SE_neo4j.Users":[find("SE_neo4j.Users","displayname")],
            "SE_neo4j.Tags":[find("SE_neo4j.Tags","tagname")]}),

 "F2": dict(cluster="F", fact="SE_neo4j.Comments",
    alt_facts=["SE_mongodb.Comments"],
    dims=["SE_neo4j.Posts","SE_neo4j.Users"], dim_alt=[],
    measures=[find("SE_neo4j.Comments","score"), find("SE_neo4j.Comments","creationdate")],
    params={"SE_neo4j.Posts":[find("SE_neo4j.Posts","id")],
            "SE_neo4j.Users":[find("SE_neo4j.Users","accountid")]}),

 "F3": dict(cluster="F", fact="SE_neo4j.Users",
    alt_facts=["SE_mysql.Users"],
    dims=["SE_neo4j.Posts","SE_neo4j.Tags"], dim_alt=["SE_neo4j.postlinks"],
    measures=[find("SE_neo4j.Users","creationdate"), find("SE_neo4j.Users","accountid")],
    params={"SE_neo4j.Posts":[find("SE_neo4j.Posts","posttypeid")],
            "SE_neo4j.Tags":[find("SE_neo4j.Tags","count")]}),

 "F4": dict(cluster="F", fact="SE_neo4j.Tags",
    alt_facts=["SE_mongodb.Posts"],
    dims=["SE_neo4j.Posts","SE_neo4j.Users"], dim_alt=[],
    measures=[find("SE_neo4j.Tags","count"), find("SE_neo4j.Tags","tagname")],
    params={"SE_neo4j.Posts":[find("SE_neo4j.Posts","parentid")],
            "SE_neo4j.Users":[find("SE_neo4j.Users","id")]}),

 "F5": dict(cluster="F", fact="SE_neo4j.postlinks",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_neo4j.Posts"], dim_alt=["SE_neo4j.Users"],
    measures=[find("SE_neo4j.postlinks","LinkTypeId"),
              find("SE_neo4j.postlinks","CreationDate")],
    params={"SE_neo4j.Posts":[find("SE_neo4j.Posts","acceptedanswerid")]}),

 # Cluster G: MongoDB-centric analytics
 "G1": dict(cluster="G", fact="SE_mongodb.Posts",
    alt_facts=["SE_mysql.postes"],
    dims=["SE_mongodb.Users","SE_mongodb.Comments"], dim_alt=["SE_neo4j.Tags"],
    measures=[find("SE_mongodb.Posts","Tags"), find("SE_mongodb.Posts","Body")],
    params={"SE_mongodb.Users":[find("SE_mongodb.Users","AboutMe")],
            "SE_mongodb.Comments":[find("SE_mongodb.Comments","Text")]}),

 "G2": dict(cluster="G", fact="SE_mongodb.Users",
    alt_facts=["SE_mysql.Users"],
    dims=["SE_mongodb.Badges","SE_mongodb.Posts"], dim_alt=[],
    measures=[find("SE_mongodb.Users","DisplayName"),
              find("SE_mongodb.Users","AboutMe")],
    params={"SE_mongodb.Badges":[find("SE_mongodb.Badges","Name"),
                                  find("SE_mongodb.Badges","Date")],
            "SE_mongodb.Posts":[find("SE_mongodb.Posts","Tags")]}),

 "G3": dict(cluster="G", fact="SE_mongodb.Comments",
    alt_facts=["SE_neo4j.Comments"],
    dims=["SE_mongodb.Users","SE_mongodb.Posts"], dim_alt=[],
    measures=[find("SE_mongodb.Comments","Text"),
              find("SE_mongodb.Comments","UserId")],
    params={"SE_mongodb.Users":[find("SE_mongodb.Users","DisplayName")],
            "SE_mongodb.Posts":[find("SE_mongodb.Posts","Id")]}),

 "G4": dict(cluster="G", fact="SE_mongodb.Badges",
    alt_facts=["SE_mysql.Flagtypes"],
    dims=["SE_mongodb.Users"], dim_alt=["SE_mysql.Users"],
    measures=[find("SE_mongodb.Badges","Name"), find("SE_mongodb.Badges","Date")],
    params={"SE_mongodb.Users":[find("SE_mongodb.Users","AboutMe")]}),

 "G5": dict(cluster="G", fact="SE_mongodb.Posts",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_neo4j.Tags","SE_mongodb.Comments"], dim_alt=["SE_mysql.Posttypes"],
    measures=[find("SE_mongodb.Posts","Body")],
    params={"SE_neo4j.Tags":[find("SE_neo4j.Tags","tagname"),
                              find("SE_neo4j.Tags","count")],
            "SE_mongodb.Comments":[find("SE_mongodb.Comments","PostId")]}),

 # Cluster H: MySQL-centric cross-store analytics
 "H1": dict(cluster="H", fact="SE_mysql.postes",
    alt_facts=["SE_mongodb.Posts"],
    dims=["SE_mysql.Votes","SE_mysql.Users"], dim_alt=["SE_mysql.Posttypes"],
    measures=[find("SE_mysql.postes","ViewCount"),
              find("SE_mysql.postes","LastEditDate")],
    params={"SE_mysql.Votes":[find("SE_mysql.Votes","VoteTypeId")],
            "SE_mysql.Users":[find("SE_mysql.Users","Reputation")]}),

 "H2": dict(cluster="H", fact="SE_mysql.Votes",
    alt_facts=["SE_mysql.postes"],
    dims=["SE_mysql.Users","SE_mysql.Votetypes"], dim_alt=["SE_mysql.postes"],
    measures=[find("SE_mysql.Votes","BountyAmount"),
              find("SE_mysql.Votes","CreationDate")],
    params={"SE_mysql.Users":[find("SE_mysql.Users","UpVotes"),
                               find("SE_mysql.Users","DownVotes")],
            "SE_mysql.Votetypes":[find("SE_mysql.Votetypes","Name")]}),

 "H3": dict(cluster="H", fact="SE_mysql.Users",
    alt_facts=["SE_neo4j.Users"],
    dims=["SE_mysql.Votes","SE_mysql.Flagtypes"], dim_alt=["SE_mongodb.Badges"],
    measures=[find("SE_mysql.Users","UpVotes"), find("SE_mysql.Users","LastAccessDate")],
    params={"SE_mysql.Votes":[find("SE_mysql.Votes","PostId")],
            "SE_mysql.Flagtypes":[find("SE_mysql.Flagtypes","Name")]}),

 "H4": dict(cluster="H", fact="SE_mysql.Votetypes",
    alt_facts=["SE_mysql.Flagtypes"],
    dims=["SE_mysql.Votes"], dim_alt=["SE_mysql.postes"],
    measures=[find("SE_mysql.Votetypes","Name")],
    params={"SE_mysql.Votes":[find("SE_mysql.Votes","CreationDate"),
                               find("SE_mysql.Votes","BountyAmount")]}),

 "H5": dict(cluster="H", fact="SE_mysql.Flagtypes",
    alt_facts=["SE_mysql.Votetypes"],
    dims=["SE_mysql.postes","SE_mysql.Users"], dim_alt=[],
    measures=[find("SE_mysql.Flagtypes","Name")],
    params={"SE_mysql.postes":[find("SE_mysql.postes","Title"),
                                find("SE_mysql.postes","ViewCount")],
            "SE_mysql.Users":[find("SE_mysql.Users","DisplayName")]}),
}

# Full 30-persona dict
PERSONAS_30 = {**PERSONAS_15, **NEW_PERSONAS}
ANALYSTS_30 = list(PERSONAS_30.keys()) + COLD

# Generate events for new personas only
def generate_new_events(seed=42, consistency=0.85):
    rng = random.Random(seed)
    new_events = []
    sid_offset = 1000  # avoid session ID collision with original 15

    def add(session, persona, rnd, itype, name, role):
        new_events.append((session, persona, rnd, itype, name, role))

    for persona, spec in NEW_PERSONAS.items():
        for rnd in range(1, 5):
            sid_offset += 1
            session = f"S{sid_offset:04d}"

            fact = spec["fact"]
            if spec["alt_facts"] and rng.random() < (1 - consistency):
                fact = rng.choice(spec["alt_facts"])
            add(session, persona, rnd, "Dataset", fact, "F")

            measures_pool = (spec["measures"] if fact == spec["fact"]
                             else (attrs_of(fact) or spec["measures"]))

            dim_pool = list(spec["dims"])
            if spec["dim_alt"] and rng.random() < 0.25:
                dim_pool = dim_pool + [rng.choice(spec["dim_alt"])]
            n_dims = min(len(dim_pool), rng.randint(1, max(1, len(dim_pool))))
            chosen_dims = rng.sample(dim_pool, n_dims)
            for d in chosen_dims:
                add(session, persona, rnd, "Dataset", d, "D")

            n_meas = min(len(measures_pool),
                         rng.randint(1, max(1, len(measures_pool))))
            for m in rng.sample(measures_pool, n_meas):
                add(session, persona, rnd, "Attribute", m, "M")

            for d in chosen_dims:
                plist = spec["params"].get(d, attrs_of(d)[:1])
                n_p = min(len(plist), rng.randint(1, max(1, len(plist))))
                for p in rng.sample(plist, n_p):
                    add(session, persona, rnd, "Attribute", p, "P")

            if rng.random() < 0.10:
                pool = attrs_of(fact)
                if pool:
                    add(session, persona, rnd, "Attribute", rng.choice(pool), "M")

    return new_events

if __name__ == "__main__":
    evs = generate_new_events()
    total = events_15 + evs
    print(f"Original 15 personas: {len(events_15)} events")
    print(f"New 15 personas:       {len(evs)} events")
    print(f"Total 30 personas:     {len(total)} events")
    print(f"Analysts: {len(ANALYSTS_30)}")
