**1. NER_vocab_10_you.txt:** This file includes all recognized entities that appear more or equal to 10 times in the youtube data and results from script 04.

**2. entities_action_you_new.json:** This file includes the entities and actions (verbs) they are combined with according to the embedding similarity resulting from script 08 and based on the **entity_verbs_you.json**.

**3. entities_you.json:** This file includes the entities in the youtube text and certain additional information such as their position. It results from the script 07 and is based on the **wordkeys_you.json** and **NER_vocab_10_you.txt**.

**4. entity_verbs_you.json:** This file includes the entities in the youtube texts and the possible verbs they can be combined with according to the constraint that the noun must according within 10 tokens of the entity. It results from the script 07.

**5. pos_nouns_you.json:** This file includes all nouns in the youtube transcripts and results from script 07.

**6. pos_verbs_you.json:** This file includes all verbs in the youtube trasncripts and results from script 07.

**7. triplets_you_lem.json:** This file includes the final triplets in a lemmatized form and results from script 08.

**8. triplets_you_new.json:** This file includes the final triplets and results from script 08.

**9. wordkeys_nouns_you.json:** This file includes the position of all nouns in the youtube transcripts and results from script 07. It was creating by combining the **wordkeys_you.json** and **pos_nouns_you.json** files.

**10. wordkeys_you.json:** This file includes the position of all words in the youtube trasncripts and results from script 07.

**11. wordkeys_verbs_you.json:** This file includes the position of all verbs in the youtube transcripts and results from script 07. It was creating by combining the **wordkeys_you.json** and **pos_verbs_you.json** files.
