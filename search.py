import torch
import os
import shutil
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import tempfile


@dataclass
class ContextNode:
    id: int
    token_id: int
    # the bonus boosts token survive beam searches
    token_score: float
    # the accumulated bonus from root to current node
    node_score: float
    # the total scores of matched phrases
    # sum of the node_score of all the output node for current node
    output_score: float
    # the distance from current node to root
    level: int
    # wether it is the end node
    is_end: bool
    # the context phrase of current node
    # the value is valid only when is_end == True
    phrase: str = ""
    # the accumulated predicted probability threshold (for kws)
    # the value is valid only when is_end == True
    accum_prob_threshold: float = 1.0
    next: Dict[int, "ContextNode"] = field(default_factory=dict)
    # jump directly to the node that has the common longest suffix
    fail: Optional["ContextNode"] = None
    # jump directly to the end node that matches the common longest suffix
    output: Optional["ContextNode"] = None


class ContextGraph:
    """The ContextGraph is modified from Aho-Corasick. A ContextGraph contains
    some phrases that are expected to have more scores during decoding and be
    more easy to survive beam searches.
    """

    def __init__(self, context_score: float, accum_prob_threshold: float = 1.0):
        """Initialize a ContextGraph with the given `context_score` and `accum_prob_threshold`.

        A root node will be created and hardcoded to -1.

        Args:
          context_score:
            The default bonus score for each token.
          accum_prob_threshold:
            The acoustic threshold (probability) to trigger the word/phrase.
        """
        self.context_score = context_score
        self.accum_prob_threshold = accum_prob_threshold
        self.num_nodes = 0
        self.root = ContextNode(
            id=self.num_nodes,
            token_id=-1,
            token_score=0,
            node_score=0,
            output_score=0,
            is_end=False,
            level=0,
        )
        self.root.fail = self.root

    def _fill_fail_output(self):
        """Fill the fail arc and the output arc for each trie node."""
        queue: deque[ContextNode] = deque()
        for token_id, node in self.root.next.items():
            node.fail = self.root
            queue.append(node)

        while queue:
            cur_node = queue.popleft()
            for token_id, node in cur_node.next.items():
                fail = cur_node.fail
                if token_id in fail.next:
                    fail = fail.next[token_id]
                else:
                    fail = fail.fail
                    while token_id not in fail.next:
                        fail = fail.fail
                        if fail.token_id == -1:  # reaching root
                            break
                    if token_id in fail.next:
                        fail = fail.next[token_id]
                node.fail = fail
                # fill the output arc
                output = node.fail
                while not output.is_end:
                    output = output.fail
                    if output.token_id == -1:  # reaching root
                        output = None
                        break
                node.output = output
                node.output_score += 0 if output is None else output.output_score
                queue.append(node)

    def build(
        self,
        token_ids: List[List[int]],
        phrases: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        accum_prob_thresholds: Optional[List[float]] = None,
    ):
        """Build the ContextGraph from a list of tokens."""
        num_phrases = len(token_ids)
        if phrases is not None:
            assert len(phrases) == num_phrases, (len(phrases), num_phrases)
        if scores is not None:
            assert len(scores) == num_phrases, (len(scores), num_phrases)
        if accum_prob_thresholds is not None:
            assert len(accum_prob_thresholds) == num_phrases, (
                len(accum_prob_thresholds),
                num_phrases,
            )

        for idx, sub_token_ids in enumerate(token_ids):
            phrase = "" if phrases is None else phrases[idx]
            score = 0.0 if scores is None else scores[idx]
            accum_prob_threshold = (
                0.0 if accum_prob_thresholds is None else accum_prob_thresholds[idx]
            )
            # use the customized score first
            context_score = self.context_score if score == 0.0 else score
            threshold = (
                self.accum_prob_threshold
                if accum_prob_threshold == 0.0
                else accum_prob_threshold
            )
            cur_node = self.root

            for i, token_id in enumerate(sub_token_ids):
                if token_id not in cur_node.next:
                    self.num_nodes += 1
                    is_end = i == len(sub_token_ids) - 1
                    node_score = cur_node.node_score + context_score
                    cur_node.next[token_id] = ContextNode(
                        id=self.num_nodes,
                        token_id=token_id,
                        token_score=context_score,
                        node_score=node_score,
                        output_score=node_score if is_end else 0,
                        is_end=is_end,
                        level=i + 1,
                        phrase=phrase if is_end else "",
                        accum_prob_threshold=threshold if is_end else 0.0,
                    )
                else:
                    # node exists, share the max score
                    token_score = max(
                        context_score, cur_node.next[token_id].token_score
                    )
                    cur_node.next[token_id].token_score = token_score
                    node_score = cur_node.node_score + token_score
                    cur_node.next[token_id].node_score = node_score
                    is_end = (
                        i == len(sub_token_ids) - 1 or cur_node.next[token_id].is_end
                    )
                    cur_node.next[token_id].output_score = node_score if is_end else 0
                    cur_node.next[token_id].is_end = is_end
                    if i == len(sub_token_ids) - 1:
                        cur_node.next[token_id].phrase = phrase
                        cur_node.next[token_id].accum_prob_threshold = threshold

                cur_node = cur_node.next[token_id]

        self._fill_fail_output()

    def forward_one_step(
        self, node: ContextNode, next_token_id: int, strict_mode: bool = True
    ) -> Tuple[float, ContextNode, ContextNode]:
        """Search the graph with given node and token.

        Args:
          node:
            The given trie node to start.
          token_id:
            The given token id.
          strict_mode:
            If the `strict_mode` is True, it can match multiple phrases simultaneously,
            and will continue to match longer phrase after matching a shorter one.
            If the `strict_mode` is False, it can only match one phrase at a time,
            when it matches a phrase, then the state will fall back to root state
            (i.e. forgetting all the history state and starting a new match). If
            the matched state have multiple outputs (node.output is not None), the
            longest phrase will be return.
            For example, if the phrases are `he`, `she` and `shell`, the query is
            `like shell`, when `strict_mode` is True, the query will match `he` and
            `she` at token `e` and `shell` at token `l`, while when `strict_mode`
            is False, the query can only match `she`(`she` is longer than `he`, so
            `she` not `he`) at token `e`.
            Note: When applying this graph for keywords spotting system, the
            `strict_mode` should be True.

        Returns:
          Return a tuple of boosting score for given node, next node and matched
          node (if any). Note: Only returns the matched node with longest phrase of
          current node, even if there are multiple matches phrases. If no phrase
          matched, the matched node is None.
        """
        next_node = None
        score = 0
        # match or reach root
        if next_token_id in node.next:  # token_id matched
            next_node = node.next[next_token_id]
            # the score of the matched path
            score = next_node.token_score
        else:  # token_id not matched
            # trace along the fail arc until it matches the token or reaching root
            next_node = node.fail
            while next_token_id not in next_node.next:
                next_node = next_node.fail
                if next_node.token_id == -1:  # reaching root
                    break
            if next_token_id in next_node.next:
                next_node = next_node.next[next_token_id]
            # the score of the fail path
            score = next_node.node_score - node.node_score
        assert next_node is not None

        # the matched node of current step, will only return the node with
        # longest phrase if there are multiple phrases matches this step
        # None if no matched phrase
        matched_node = (
            next_node
            if next_node.is_end
            else (next_node.output if next_node.output is not None else None)
        )
        if not strict_mode and next_node.output_score != 0:
            # output_score != 0 means at least one phrase matched
            assert matched_node is not None
            if next_node.is_end:
                output_score = next_node.node_score
            elif next_node.output is None:
                output_score = next_node.node_score
            else:
                output_score = next_node.output.node_score
            return (
                score + output_score - next_node.node_score,
                self.root,
                matched_node,
            )
        assert (next_node.output_score != 0 and matched_node is not None) or (
            next_node.output_score == 0 and matched_node is None
        ), (next_node.output_score, matched_node)
        return (score + next_node.output_score, next_node, matched_node)

    def is_matched(self, node: ContextNode) -> Tuple[bool, ContextNode]:
        """Whether current node matches any phrase, i.e. current node is the
        end node or the output of current node is not None."""
        if node.is_end:
            return True, node
        else:
            if node.output is not None:
                return True, node.output
            return False, None

    def visualize(
        self,
        title: Optional[str] = None,
        filename: Optional[str] = "",
        symbol_table: Optional[Dict[int, str]] = None,
    ):
        """Visualize a ContextGraph via graphviz.

        Args:
           title:
              Title to be displayed in the image.
           filename:
              Filename to (optionally) save to, e.g. 'foo.png', 'foo.svg',
              'foo.png'  (must have a suffix that graphviz understands).
           symbol_table:
              Map the token ids to symbols.
        Returns:
          A Diagraph from grahpviz.
        """
        import graphviz

        graph_attr = {
            "rankdir": "LR",
            "size": "8.5,11",
            "center": "1",
            "orientation": "Portrait",
            "ranksep": "0.4",
            "nodesep": "0.25",
        }
        if title is not None:
            graph_attr["label"] = title
        default_node_attr = {"shape": "circle", "style": "bold", "fontsize": "14"}
        final_state_attr = {"shape": "doublecircle", "style": "bold", "fontsize": "14"}

        dot = graphviz.Digraph(name="Context Graph", graph_attr=graph_attr)
        seen = set()
        queue = deque()
        queue.append(self.root)
        dot.node("0", label="0", **default_node_attr)
        dot.edge("0", "0", color="red")
        seen.add(0)

        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.id not in seen:
                    node_score = f"{node.node_score:.2f}".rstrip("0").rstrip(".")
                    output_score = f"{node.output_score:.2f}".rstrip("0").rstrip(".")
                    label = f"{node.id}/({node_score}, {output_score})"
                    if node.is_end:
                        dot.node(str(node.id), label=label, **final_state_attr)
                    else:
                        dot.node(str(node.id), label=label, **default_node_attr)
                    seen.add(node.id)
                weight = f"{node.token_score:.2f}".rstrip("0").rstrip(".")
                label = str(token) if symbol_table is None else symbol_table[token]
                dot.edge(str(current_node.id), str(node.id), label=f"{label}/{weight}")
                dot.edge(str(node.id), str(node.fail.id), color="red")
                if node.output is not None:
                    dot.edge(str(node.id), str(node.output.id), color="green")
                queue.append(node)

        if filename:
            _, extension = os.path.splitext(filename)
            if extension == "" or extension[0] != ".":
                raise ValueError(
                    "Filename needs to have a suffix like .png, .pdf, .svg: {}".format(
                        filename
                    )
                )
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = dot.render(
                    filename="temp",
                    directory=tmp_dir,
                    format=extension[1:],
                    cleanup=True,
                )
                shutil.move(temp_fn, filename)

        return dot


@dataclass
class Hypothesis:
    ys: List[int]  # the predicted tokens so far
    log_prob: torch.Tensor  # the log probability of ys
    # store every probability from predicted tokens
    accum_probs: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    context_node: Optional[ContextNode] = None
    num_tailing_blanks: int = 0

    @property
    def key(self) -> str:
        return "_".join(map(str, self.ys))


class HypothesisDict(dict[str, Hypothesis]):
    def add(self, hyp: Hypothesis):
        key = hyp.key
        if key in self:
            old_hyp = self[key]
            old_hyp.log_prob = torch.logaddexp(old_hyp.log_prob, hyp.log_prob)
        else:
            self[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        if length_norm:
            return max(self.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self.values(), key=lambda hyp: hyp.log_prob)


@dataclass
class KeywordResult:
    timestamps: List[int]
    token_ids: List[int]
    phrase: str


class KeywordSearcher:
    def __init__(
        self,
        keywords: List[str],
        blank_id: int,
        jumped_ids: List[int],
        tokens2ids: dict[str, int],
        beam: int = 4,
        blank_penalty: float = 0,
        min_num_tailing_blanks: int = 0,
        context_size: int = 1,
    ):
        self.jumped_ids = jumped_ids
        self.blank_id = blank_id
        self.tokens2ids = tokens2ids
        self.keyword_graph = self.build_keyword_graph(keywords)
        self.beam = beam
        self.blank_penalty = blank_penalty
        self.min_num_tailing_blanks = min_num_tailing_blanks
        self.context_size = context_size

    def build_keyword_graph(
        self,
        keywords: List[str],
        default_keywords_score=4,
        default_accum_prob_threshold=0.1,
    ):
        token_ids = []
        phrases = []
        keywords_scores = []
        keywords_thresholds = []

        for keyword_cfg in keywords:
            keyword = keyword_cfg["keyword"].strip().upper()
            score = keyword_cfg["score"]
            threshold = keyword_cfg["threshold"]
            tmp_ids = []
            for k in keyword:
                if k in self.tokens2ids:
                    tmp_ids.append(self.tokens2ids[k])
                else:
                    tmp_ids = []
                    break
            if tmp_ids:
                phrases.append(keyword)
                token_ids.append(tmp_ids)
                keywords_scores.append(score)
                keywords_thresholds.append(threshold)

        keywords_graph = ContextGraph(
            context_score=default_keywords_score,
            accum_prob_threshold=default_accum_prob_threshold,
        )
        keywords_graph.build(
            token_ids=token_ids,
            phrases=phrases,
            scores=keywords_scores,
            accum_prob_thresholds=keywords_thresholds,
        )
        return keywords_graph

    def calc_sample_splits_ids(self, states: List[HypothesisDict]):
        num_hyps = torch.tensor([0] + [len(h) for h in states])
        sample_splits = num_hyps.cumsum(0).long()
        sample_ids = []
        for i, c in enumerate(num_hyps[1:]):
            sample_ids += [i] * c
        sample_ids = torch.tensor(sample_ids).long()
        return sample_splits, sample_ids

    def check_matching(
        self, states: List[HypothesisDict], check_num_tailing_blanks: bool = False
    ):
        num_states = len(states)
        answers: List[KeywordResult] = [None for _ in range(num_states)]
        for i in range(num_states):
            # get the sequence that has the biggest probability
            top_hyp = states[i].get_most_probable(length_norm=True)
            matched, matched_node = self.keyword_graph.is_matched(top_hyp.context_node)
            if not matched:
                continue
            accum_prob = sum(top_hyp.accum_probs[-matched_node.level :])
            accum_prob /= matched_node.level
            if not accum_prob >= matched_node.accum_prob_threshold:
                continue
            if check_num_tailing_blanks:
                if not top_hyp.num_tailing_blanks >= self.min_num_tailing_blanks:
                    continue

            answers[i] = KeywordResult(
                token_ids=top_hyp.ys[-matched_node.level :],
                timestamps=top_hyp.timestamps[-matched_node.level :],
                phrase=matched_node.phrase,
            )
        return answers

    @torch.no_grad()
    def search_one_step(
        self,
        predicted_logits: torch.Tensor,
        states: List[HypothesisDict],
        timestep: int,
    ):
        device = predicted_logits.device

        if len(states) == 0:
            # initialize states
            # every sample has a HypothesisDict
            # every HypothesisDict save multiple results
            batch_size = predicted_logits.size(0)
            states = [HypothesisDict() for _ in range(batch_size)]
            for i in range(batch_size):
                states[i].add(
                    Hypothesis(
                        ys=[-1] * (self.context_size - 1) + [self.blank_id],
                        log_prob=torch.zeros(1).float(),
                        accum_probs=[],
                        context_node=self.keyword_graph.root,
                        timestamps=[],
                        num_tailing_blanks=0,
                    )
                )

        batch_size = len(states)
        answers: List[List[KeywordResult]] = [[] for _ in range(batch_size)]
        last_states = [list(s.values()) for s in states]
        updated_states = [HypothesisDict() for _ in range(batch_size)]

        if self.blank_penalty != 0:
            predicted_logits[:, self.blank_id] -= self.blank_penalty

        # get last predicted `log_prob`
        last_log_probs = torch.cat(
            [
                hyp.log_prob.reshape(1, 1).to(device)
                for hyps in last_states
                for hyp in hyps
            ]
        ).to(device)
        probs = predicted_logits.softmax(dim=-1)
        log_probs = probs.log()
        # real probability = accumulated probability * predicted probability
        log_probs.add_(last_log_probs)

        vocab_size = log_probs.size(-1)
        sample_splits = torch.tensor([0] + [len(h) for h in states]).cumsum(0).long()
        scaled_sample_splits = sample_splits * vocab_size
        probs = probs.reshape(-1)
        log_probs = log_probs.reshape(-1)

        for i in range(batch_size):
            # get `log_probs` and `probs` from one of samples
            s, e = scaled_sample_splits[i : i + 2]
            topk_log_probs, topk_indexes = log_probs[s:e].topk(self.beam)
            hyp_probs = probs[s:e].tolist()

            topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
            topk_token_ids = (topk_indexes % vocab_size).tolist()

            # `context_node` is a node in the `self.keywords_graph`
            for k in range(len(topk_hyp_indexes)):
                hyp: Hypothesis = last_states[i][topk_hyp_indexes[k]]
                new_ys = hyp.ys[:]
                new_timestamps = hyp.timestamps[:]
                new_accum_probs = hyp.accum_probs[:]
                new_context_node = hyp.context_node
                new_num_tailing_blanks = hyp.num_tailing_blanks + 1
                new_token_id = topk_token_ids[k]
                context_score = 0

                if new_token_id not in self.jumped_ids:  # jump nonsense tokens
                    new_ys.append(new_token_id)
                    new_timestamps.append(timestep)
                    new_accum_probs.append(hyp_probs[topk_indexes[k]])
                    # check wether `new_token_id` is matching any keywords
                    context_score, new_context_node, _ = (
                        self.keyword_graph.forward_one_step(
                            hyp.context_node, new_token_id
                        )
                    )
                    new_num_tailing_blanks = 0
                    # `token_id == -1` indicates no keyword matching
                    if new_context_node.token_id == -1:
                        new_ys[-self.context_size :] = [-1] * (
                            self.context_size - 1
                        ) + [self.blank_id]

                # boost score for the matched node
                new_log_prob = topk_log_probs[k] + context_score
                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamps=new_timestamps,
                    accum_probs=new_accum_probs,
                    context_node=new_context_node,
                    num_tailing_blanks=new_num_tailing_blanks,
                )
                # if `new_hyp` is existed, then jump
                updated_states[i].add(new_hyp)

            # check wether there are matched keywords
            ans = self.check_matching([updated_states[i]], True)[0]
            if ans is not None:
                answers[i].append(ans)
                # remove all sequences and restart matching
                updated_states[i] = HypothesisDict()
                updated_states[i].add(
                    Hypothesis(
                        ys=[-1] * (self.context_size - 1) + [self.blank_id],
                        log_prob=torch.zeros(1).float(),
                        context_node=self.keyword_graph.root,
                        timestamps=[],
                        accum_probs=[],
                    )
                )

        return answers, updated_states

    def final_search(self, final_states: List[HypothesisDict]):
        # keywords at the end of sequences don't need to care the number of blank chars
        # so check again to ensure not missing the final keywords
        final_answers = self.check_matching(
            final_states, check_num_tailing_blanks=False
        )
        answers = [[] if ans is None else [ans] for ans in final_answers]
        return answers


@dataclass
class ASRResults:
    timestamps: List[List[int]]
    token_ids: List[List[int]]
    texts: Union[List[str], None]


class BeamSearcher:
    def __init__(
        self,
        hotwords: List[dict],
        blank_id: int,
        jumped_ids: List[int],
        token_table: dict,
        beam: int = 4,
        temperature: float = 1.0,
        blank_penalty: float = 0,
        context_size: int = 1,
    ):
        self.jumped_ids = jumped_ids
        self.blank_id = blank_id
        self.token_table = token_table
        self.beam = beam
        self.temperature = temperature
        self.blank_penalty = blank_penalty
        self.context_size = context_size
        self.context_graph = None

        if len(hotwords) > 0:
            self.context_graph = self.build_context_graph(hotwords)

    def build_context_graph(
        self,
        hotwords: List[dict],
        default_keywords_score=4,
    ):
        token_ids = []
        phrases = []
        hotwords_scores = []

        for keyword_cfg in hotwords:
            keyword = keyword_cfg["keyword"].strip().upper()
            score = keyword_cfg["score"]
            tmp_ids = []
            for k in keyword:
                if k in self.token_table:
                    tmp_ids.append(self.token_table[k])
                else:
                    tmp_ids = []
                    break
            if tmp_ids:
                phrases.append(keyword)
                token_ids.append(tmp_ids)
                hotwords_scores.append(score)

        context_graph = ContextGraph(context_score=default_keywords_score)
        context_graph.build(
            token_ids=token_ids, phrases=phrases, scores=hotwords_scores
        )
        return context_graph

    def calc_sample_splits_ids(self, states: List[HypothesisDict]):
        num_hyps = torch.tensor([0] + [len(h) for h in states])
        sample_splits = num_hyps.cumsum(0).long()
        sample_ids = []
        for i, c in enumerate(num_hyps[1:]):
            sample_ids += [i] * c
        sample_ids = torch.tensor(sample_ids).long()
        return sample_splits, sample_ids

    @torch.no_grad()
    def search_one_step(
        self,
        predicted_logits: torch.Tensor,
        states: List[HypothesisDict],
        timestep: int,
    ):
        device = predicted_logits.device

        if len(states) == 0:
            # initialize states
            # every sample has a HypothesisDict
            # every HypothesisDict save multiple results
            batch_size = predicted_logits.size(0)
            states = [HypothesisDict() for _ in range(batch_size)]
            for i in range(batch_size):
                states[i].add(
                    Hypothesis(
                        ys=[-1] * (self.context_size - 1) + [self.blank_id],
                        log_prob=torch.zeros(1).float(),
                        context_node=(
                            None
                            if self.context_graph is None
                            else self.context_graph.root
                        ),
                        timestamps=[],
                    )
                )

        batch_size = len(states)
        last_states = [list(s.values()) for s in states]
        updated_states = [HypothesisDict() for _ in range(batch_size)]

        if self.blank_penalty != 0:
            predicted_logits[:, self.blank_id] -= self.blank_penalty

        # get last predicted `log_prob`
        last_log_probs = torch.cat(
            [
                hyp.log_prob.reshape(1, 1).to(device)
                for hyps in last_states
                for hyp in hyps
            ]
        ).to(device)
        log_probs = (predicted_logits / self.temperature).log_softmax(dim=-1)
        # real probability = accumulated probability * predicted probability
        log_probs.add_(last_log_probs)

        vocab_size = log_probs.size(-1)
        sample_splits = torch.tensor([0] + [len(h) for h in states]).cumsum(0).long()
        scaled_sample_splits = sample_splits * vocab_size
        log_probs = log_probs.reshape(-1)

        for i in range(batch_size):
            # get all searched `log_probs` from one of samples
            s, e = scaled_sample_splits[i : i + 2]
            topk_log_probs, topk_indexes = log_probs[s:e].topk(self.beam)

            topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
            topk_token_ids = (topk_indexes % vocab_size).tolist()

            # `context_node` is a node in the `self.keywords_graph`
            for k in range(len(topk_hyp_indexes)):
                hyp: Hypothesis = last_states[i][topk_hyp_indexes[k]]
                new_ys = hyp.ys[:]
                new_timestamps = hyp.timestamps[:]
                new_context_node = (
                    None if self.context_graph is None else hyp.context_node
                )
                new_token_id = topk_token_ids[k]
                context_score = 0

                if new_token_id not in self.jumped_ids:  # jump nonsense tokens
                    new_ys.append(new_token_id)
                    new_timestamps.append(timestep)

                    if self.context_graph is not None:
                        # check wether `new_token_id` is matching any keywords
                        context_score, new_context_node, _ = (
                            self.context_graph.forward_one_step(
                                hyp.context_node, new_token_id
                            )
                        )

                # boost score for the matched node
                new_log_prob = topk_log_probs[k] + context_score
                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamps=new_timestamps,
                    context_node=new_context_node,
                )
                # if `new_hyp` is existed, then jump
                updated_states[i].add(new_hyp)

        return updated_states

    def get_best_answers(
        self, final_states: List[HypothesisDict], ids2tokens: dict[int, str] = None
    ):
        if self.context_graph is not None:
            tmp_states = [HypothesisDict() for _ in range(len(final_states))]
            for i, hyps in enumerate(final_states):
                for hyp in hyps.values():
                    tmp_states[i].add(
                        Hypothesis(
                            ys=hyp.ys,
                            log_prob=hyp.log_prob - hyp.context_node.node_score,
                            timestamps=hyp.timestamps,
                            context_node=self.context_graph.root,
                        )
                    )
            final_states = tmp_states

        best_hyps = [s.get_most_probable(length_norm=True) for s in final_states]
        token_ids = [h.ys[self.context_size :] for h in best_hyps]
        timestamps = [h.timestamps for h in best_hyps]
        texts = None
        if ids2tokens is not None:
            texts = []
            for tokens in token_ids:
                texts.append("".join([ids2tokens[idx] for idx in tokens]))

        return ASRResults(token_ids=token_ids, timestamps=timestamps, texts=texts)
