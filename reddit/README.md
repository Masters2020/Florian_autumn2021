**1. act_nouns_red.json:** This file includes the entity-verbs pairs in the reddit texts and the possible nouns they can be combined with according to the constraint that the noun must according within 10 tokens of the verb. It results from the script 07

**2. entities_action_red.json:** This file includes the entities and actions (verbs) they are combined with according to the embedding similarity resulting from script 08 and based on the **entity_verbs_red.json**.

**3. entities_red.json:** This file includes the entities in the reddit text and certain additional information such as their position. It results from the script 07.

**4. entity_verbs_red.json:** This file includes the entities in the reddit texts and the possible verbs they can be combined with according to the constraint that the noun must according within 10 tokens of the entity. It results from the script 07

**5. pos_nouns_red.json:** This file includes all nouns in the reddit posts. Combined with the wordkeys dictionary, a dictionary can be created including the position of the nouns in the text. It results from script 07

**6. pos_verbs_red.json:** This file includes all verbs in the reddit posts. Combined with the wordkeys dictionary, a dictionary can be created including the position of the verbs in the text. It results from script 07

**7. triplets_red_lem.json:** This file includes the final triplets in a lemmatized form and results from script 08

**8. triplets_red_new.json:** This file includes the final triplets and results from script 08

**9. wordkeys_red.json:** This file includes the position of all words in the reddit posts and results from script 07
