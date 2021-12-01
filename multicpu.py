# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  uniprot_accession_ids = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      uniprot_accession_ids.append(
          identifiers.uniprot_accession_id.encode('utf-8'))
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_uniprot_accession_identifiers'] = np.array(
      uniprot_accession_ids, dtype=np.object_)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    with open(msa_out_path, 'r') as f:
      result = {msa_format: f.read()}
  return result


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    jackhmmer_uniref90_process, jupdir = self.jackhmmer_uniref90_runner.query(input_fasta_path)
    jackhmmer_mgnify_process, jmpdir = self.jackhmmer_mgnify_runner.query(input_fasta_path)
    hhblits_bfd_uniclust_process, hbudir = self.hhblits_bfd_uniclust_runner.query(input_fasta_path)

    for p in [jackhmmer_uniref90_process, jackhmmer_mgnify_process, hhblits_bfd_uniclust_process]:
      p.wait()
    jackhmmer_uniref90_result = open(os.path.join(jupdir, 'output.sto'), 'r').read()
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    with open(uniref90_out_path, 'w') as f:
      f.write(jackhmmer_uniref90_result)

    jackhmmer_mgnify_result = open(os.path.join(jmpdir, 'output.sto'), 'r').read()
    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    with open(mgnify_out_path, 'w') as f:
      f.write(jackhmmer_mgnify_result)
    
    '''
    if self._use_small_bfd:
      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
      jackhmmer_small_bfd_result = run_msa_tool(
          self.jackhmmer_small_bfd_runner, input_fasta_path, bfd_out_path,
          'sto', self.use_precomputed_msas)
      bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
    else:
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
      hhblits_bfd_uniclust_result = run_msa_tool(
          self.hhblits_bfd_uniclust_runner, input_fasta_path, bfd_out_path,
          'a3m', self.use_precomputed_msas)
      bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])
    '''
    hhblits_bfd_uniclust_result = open(os.path.join(hbudir, 'output.a3m'), 'r').read()
    bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result)

    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    with open(bfd_out_path, 'w') as f:
      f.write( hhblits_bfd_uniclust_result)

    msa_for_templates = jackhmmer_uniref90_result['sto']
    msa_for_templates = parsers.truncate_stockholm_msa(
        msa_for_templates, max_sequences=self.uniref_max_hits)
    msa_for_templates = parsers.deduplicate_stockholm_msa(
        msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)

    if self.template_searcher.input_format == 'sto':
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
    uniref90_msa = uniref90_msa.truncate(max_seqs=self.uniref_max_hits)
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
    mgnify_msa = mgnify_msa.truncate(max_seqs=self.mgnify_max_hits)

    pdb_template_hits = self.template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_sequence)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        hits=pdb_template_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}

import time
import pickle
def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline,
    is_prokaryote):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  if is_prokaryote is None:
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
  else:
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir,
        is_prokaryote=is_prokaryote)
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)


  
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
import numpy as np

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_list(
    'is_prokaryote_list', None, 'Optional for multimer system, not used by the '
    'single chain system. This list should contain a boolean for each fasta '
    'specifying true where the target complex is from a prokaryote, and false '
    'where it is not, or where the origin is unknown. These values determine '
    'the pairing method for the MSA.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk. WARNING: This will not check '
                     'if the sequence, database or configuration have changed.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')
def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniclust30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)
  _check_flag('pdb_seqres_database_path', 'model_preset',
              should_be_set=run_multimer_system)
  _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  # Check that is_prokaryote_list has same number of elements as fasta_paths,
  # and convert to bool.
  if FLAGS.is_prokaryote_list:
    if len(FLAGS.is_prokaryote_list) != len(FLAGS.fasta_paths):
      raise ValueError('--is_prokaryote_list must either be omitted or match '
                       'length of --fasta_paths.')
    is_prokaryote_list = []
    for s in FLAGS.is_prokaryote_list:
      if s in ('true', 'false'):
        is_prokaryote_list.append(s == 'true')
      else:
        raise ValueError('--is_prokaryote_list must contain comma separated '
                         'true or false values.')
  else:  # Default is_prokaryote to False.
    is_prokaryote_list = [False] * len(fasta_names)

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas)

  if run_multimer_system:
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_names))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    is_prokaryote = is_prokaryote_list[i] if run_multimer_system else None
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        is_prokaryote=is_prokaryote)

  if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
  ])

  app.run(main)