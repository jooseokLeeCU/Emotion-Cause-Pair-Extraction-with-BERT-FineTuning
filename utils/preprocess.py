from sklearn.utils import shuffle
import json

def sample_to_SQuAD(sample):
  '''
  Description: This function transform a sample into SQuAD data format.
               The sample is a single conversation consists of several utterances.
               The format of the output follows the SQuAD data format, namely context, question, and answer.
               There could be multiple outputs out of a single sample.

  Args:
     sample (List[dict]): A sample from the original data

  Output:
     contexts_sample (List[str]): The context part of a sample
     questions_sample (List[str]): The question part of a sample
     answers_sample (List[str]): The answer part of a sample

  '''

  contexts_sample = []
  questions_sample = []
  answers_sample = []

  convs = sample['conversation'] # Conversation part of a sample
  ecps = sample['emotion-cause_pairs'] # Emotion-cause_pairs part of a sample
  target_idxs = [int(x[0].split('_')[0])-1 for x in ecps] # indices of utterance in the conversation that has the emotion-cause_pairs
  '''
  # Negative examples (Utterances without emotion-cause_pairs)
  for i, conv in enumerate(convs):
    if i not in target_idxs: # Make a negative example
      question = conv['text']
      context = " ".join([x['text'] for x in convs[:i+1]])
      answer = {'text': '', 'answer_start': 0, 'answer_end': 0}
      contexts_sample.append(context)
      questions_sample.append(question)
      answers_sample.append(answer)
  '''
  # Positive examples (Utterances with emotion-cause_pairs)
  for ecp in ecps:
    ecp_idx = int(ecp[0].split('_')[0])-1

    if (ecp_idx in target_idxs) and (int(ecp[0].split('_')[0])>=int(ecp[1].split('_')[0])): # Get rid of cases where emotions come first than the corresponding causes
      question = convs[ecp_idx]['text']
      context = " ".join([x['text'] for x in convs[:ecp_idx+1]])
      answer_txt = ecp[1].split('_')[1]
      answer_start = context.index(answer_txt)
      answer_end = answer_start + len(answer_txt)
      answer = {'text': [answer_txt], 'answer_start': [answer_start]}
      contexts_sample.append(context)
      questions_sample.append(question)
      answers_sample.append(answer)

  return contexts_sample, questions_sample, answers_sample


def SQuAD_format_transformation(origin_data, random_state=42):
  '''
  Description: This function transforms the origin_data into SQuAD data format.
               The original data is a set of conversations that are subdivided into utterances.
               For each conversation, emotion-cause_pairs, which shows the casual relationship between utterances within it, are also defined.
               An example of the original data is as follows:
                   conversation ID: 1
                   covnersation: [utterance 1, utterance 2, utterance 3, ..., utterence n]
                   emotion-cause_pairs: [[u3(Joy), utterance 1 (subtext), utterance 2 (subtext)], [u4(Sad), utterance 1 (subtext), utterance 3 (subtext)], ...]
               The output format of the function is SQuAD format. That is, it consists of context, question, and the corresponding answer.
               In SQuAD format, the instance is based on an utterance level instead of conversation level. The question part is the target utterance of the given emotion (e.g., u3(Joy)),
               and the context part is the previous utterances including the target utterance itself in the conversation (e.g., utterance 1, utterance 2).
               The answer part is the specific cause(s) of the given emotion (e.g., utterance 1 (subtext), utterance 2 (subtext)).
               An example of the output data is as follows:
                   context: utterance 1, utterance 2, utterance 3 (concatenated)
                   question: utterance 3
                   answer: [{'text': utterance 1 (subtext), 'answer_start': 3, 'answer_end': 10}}]
               Finally, we also added negative examples in the output. An example of a negative example is as follows:
                   context: utterance 1, utterance 2
                   question: utterance 2
                   answer: []

  Args:
     origin_data (List[dict]): The original data from the semEval project

  Output:
     contexts (List[str]): The context part of the SQuAD dataset (In this case, the previous utterance in a given conversation including the target utterance itself)
     questions (List[str]): The question part of the SQuAD dataset (In this case, the target utterance)
     answers (List[dic]): The answer part of the SQuAD dataset (In this case, the subtext, which is the direct cause of the given emotion, in the corresponding context)
  '''
  from sklearn.utils import shuffle

  contexts = []
  questions = []
  answers = []

  for sample in origin_data:
    contexts_sample, questions_sample, answers_sample = sample_to_SQuAD(sample)
    contexts = contexts + contexts_sample
    questions = questions + questions_sample
    answers = answers + answers_sample

  contexts, questions, answers = shuffle(contexts, questions, answers, random_state = random_state)

  return contexts, questions, answers

def process_json_file(file_path):
    try:
        conversation_ids = []
        conversations = []
        emotion_labels = []

        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Process the data (this is just a simple example, adjust as needed)
        for conversation in data:
            #print(f"Conversation ID: {conversation['conversation_ID']}")
            conversation_ids.append(conversation['conversation_ID'])

            utterances = []
            emotions = []
            # Iterate over each utterance in the conversation
            for utterance in conversation['conversation']:
              utterances.append(utterance['text'])
              emotions.append(utterance['emotion'])

            conversations.append(utterances)
            emotion_labels.append(emotions)

            #for utterance in conversation['conversation']:
            #    print(f"  Utterance ID: {utterance['utterance_ID']}, Speaker: {utterance['speaker']}, Emotion: {utterance['emotion']}")
            #    print(f"    Text: {utterance['text']}")

            #print("\nEmotion-Cause Pairs:")
            #for pair in conversation['emotion-cause_pairs']:
            #    print(f"  {pair}")
        return conversation_ids, conversations, emotion_labels
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")