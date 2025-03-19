import pandas as pd

from typing import Dict, List
from tqdm import tqdm

from llm_annotator import utils


def mark_ineligible_rows(model_list: List[str],
                         feature_dict: Dict,
                         transcript_df: pd.DataFrame,
                         min_len):
    # Create separate dfs for individual features
    atn_feature_dfs = {feature_name: transcript_df.copy() for feature_name in feature_dict.keys()}

    # Filter out ineligible rows
    eligible_rows = transcript_df[(transcript_df['role'] == 'student') &
                                  (transcript_df['dialogue'].str.split().str.len() >= min_len)]
    ineligible_rows = transcript_df.index.difference(eligible_rows.index)

    # Mark ineligible rows with Nones
    for model_name in model_list:
        for feature_name in feature_dict:
            for idx in ineligible_rows:
                atn_feature_dfs[feature_name].at[idx, model_name] = None

    return eligible_rows, atn_feature_dfs


@utils.component("process_observations")
def process_observations(transcript_df: pd.DataFrame,
                         model_list: List[str],
                         feature_dict: Dict,
                         obs_list: List[str] = None,
                         if_context: bool = False,
                         fwd_window: int = 0,
                         bwd_window: int = 0,
                         min_len: int = 6,
                         **kwargs) -> Dict[str, pd.DataFrame]:

    atn_df = transcript_df
    # Create a dictionary to store results per feature
    eligible_rows, atn_feature_dfs = mark_ineligible_rows(model_list=model_list,
                                                          feature_dict=feature_dict,
                                                          transcript_df=transcript_df,
                                                          min_len=min_len)

    # Prepare schemas and parser once (moved outside loop)
    # To-be implemented

    # Group data by observation ID for faster context window construction
    obs_groups = {}
    if if_context:
        for obs_id in obs_list:
            obs_groups[obs_id] = atn_df[atn_df['obsid'] == obs_id].index.tolist()

    # Process eligible rows
    annotated_set = set()

    for idx, row in tqdm(eligible_rows.iterrows(), desc="Processing annotations", total=len(eligible_rows)):
        if row['obsid'] not in annotated_set:
            print(f"Annotating obs {row['obsid']}.")
            annotated_set.add(row['obsid'])

        # Build context window if requested
        window = ""
        if if_context:
            window = self._build_context_window(row, atn_df, obs_groups, fwd_window, bwd_window)

        # Get annotations from each model
        for model_name, chain in self.chains.items():
            try:
                annotation_results = self._annotate_utterance(
                    chain, row['dialogue'], annotation_format, window
                )

                # Update all features at once for this row and model
                for feature_name in self.features:
                    feature_dfs[feature_name].at[idx, model_name] = annotation_results[feature_name]
            except KeyboardInterrupt:
                self._save_results(feature_dfs=feature_dfs,
                                  models=list(self.chains.keys()),
                                  obs_list=obs_list,
                                  if_context=if_context,
                                  fwd_window=fwd_window,
                                  bwd_window=bwd_window,
                                  **kwargs)
                return feature_dfs
            except Exception as e:
                # Re-raise KeyboardInterrupt to be caught by outer handler
                self._save_results(feature_dfs=feature_dfs,
                                  models=list(self.chains.keys()),
                                  obs_list=obs_list,
                                  if_context=if_context,
                                  fwd_window=fwd_window,
                                  bwd_window=bwd_window,
                                  **kwargs)
                return feature_dfs
        if idx % 10 ==0:
            self._save_results(feature_dfs=feature_dfs,
                                  models=list(self.chains.keys()),
                                  obs_list=obs_list,
                                  if_context=if_context,
                                  fwd_window=fwd_window,
                                  bwd_window=bwd_window,
                                  **kwargs)

    # Save results
    self._save_results(feature_dfs=feature_dfs,
                      models=list(self.chains.keys()),
                      obs_list=obs_list,
                      if_context=if_context,
                      fwd_window=fwd_window,
                      bwd_window=bwd_window,
                      **kwargs)
    return feature_dfs