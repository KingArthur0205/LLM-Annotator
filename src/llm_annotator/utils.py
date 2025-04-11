import inspect
import functools
import catalogue
import os
import datetime
import json

from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class BatchRequestCounts:
    """Class to represent request counts in a batch job"""
    # OpenAI fields
    total: int = 0
    completed: int = 0
    failed: int = 0

    # Anthropic fields
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


@dataclass
class Batch:
    """Class to represent an API batch job with all its metadata for both OpenAI and Anthropic"""
    # Common fields
    id: str
    status: str  # OpenAI uses 'status', Anthropic uses 'processing_status'
    created_at: Any  # Could be int (OpenAI) or str (Anthropic)
    expires_at: Any  # Could be int (OpenAI) or str (Anthropic)
    request_counts: BatchRequestCounts
    stored_at: str  # Local timestamp when the metadata was stored

    # OpenAI specific fields
    object: Optional[str] = None
    endpoint: Optional[str] = None
    errors: Optional[List[Dict[str, Any]]] = None
    input_file_id: Optional[str] = None
    completion_window: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    in_progress_at: Optional[Any] = None  # int for OpenAI
    completed_at: Optional[Any] = None  # int for OpenAI
    failed_at: Optional[Any] = None  # int for OpenAI
    expired_at: Optional[Any] = None  # int for OpenAI
    metadata: Optional[Dict[str, Any]] = None

    # Anthropic specific fields
    type: Optional[str] = None
    processing_status: Optional[str] = None  # Maps to status
    ended_at: Optional[str] = None
    cancel_initiated_at: Optional[str] = None
    results_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Batch':
        """Create a Batch instance from a dictionary"""
        # Determine if this is OpenAI or Anthropic format based on keys
        is_anthropic = 'type' in data or 'processing_status' in data

        # Handle request counts appropriately
        request_counts_data = data.get('request_counts', {})
        if request_counts_data is None:
            request_counts_data = {}

        request_counts = BatchRequestCounts()

        # Set all possible fields (OpenAI)
        if 'total' in request_counts_data:
            request_counts.total = request_counts_data.get('total', 0)
        if 'completed' in request_counts_data:
            request_counts.completed = request_counts_data.get('completed', 0)
        if 'failed' in request_counts_data:
            request_counts.failed = request_counts_data.get('failed', 0)

        # Set all possible fields (Anthropic)
        if 'processing' in request_counts_data:
            request_counts.processing = request_counts_data.get('processing', 0)
        if 'succeeded' in request_counts_data:
            request_counts.succeeded = request_counts_data.get('succeeded', 0)
        if 'errored' in request_counts_data:
            request_counts.errored = request_counts_data.get('errored', 0)
        if 'canceled' in request_counts_data:
            request_counts.canceled = request_counts_data.get('canceled', 0)
        if 'expired' in request_counts_data:
            request_counts.expired = request_counts_data.get('expired', 0)

        # Use the appropriate status field
        status = data.get('status', '')
        if is_anthropic and not status:
            status = data.get('processing_status', '')

        # Create the Batch object with all fields
        return cls(
            # Common fields
            id=data.get('id', ''),
            status=status,
            created_at=data.get('created_at', 0),
            expires_at=data.get('expires_at', 0),
            request_counts=request_counts,
            stored_at=data.get('stored_at', ''),

            # OpenAI specific fields
            object=data.get('object'),
            endpoint=data.get('endpoint'),
            errors=data.get('errors'),
            input_file_id=data.get('input_file_id'),
            completion_window=data.get('completion_window'),
            output_file_id=data.get('output_file_id'),
            error_file_id=data.get('error_file_id'),
            in_progress_at=data.get('in_progress_at'),
            completed_at=data.get('completed_at'),
            failed_at=data.get('failed_at'),
            expired_at=data.get('expired_at'),
            metadata=data.get('metadata'),

            # Anthropic specific fields
            type=data.get('type'),
            processing_status=data.get('processing_status'),
            ended_at=data.get('ended_at'),
            cancel_initiated_at=data.get('cancel_initiated_at'),
            results_url=data.get('results_url')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Batch object to a dictionary"""
        return {
            'id': self.id,
            'object': self.object,
            'endpoint': self.endpoint,
            'errors': self.errors,
            'input_file_id': self.input_file_id,
            'completion_window': self.completion_window,
            'status': self.status,
            'output_file_id': self.output_file_id,
            'error_file_id': self.error_file_id,
            'created_at': self.created_at,
            'in_progress_at': self.in_progress_at,
            'expires_at': self.expires_at,
            'completed_at': self.completed_at,
            'failed_at': self.failed_at,
            'expired_at': self.expired_at,
            'request_counts': {
                'total': self.request_counts.total,
                'completed': self.request_counts.completed,
                'failed': self.request_counts.failed
            },
            'metadata': self.metadata,
            'stored_at': self.stored_at
        }


components = catalogue.create("llm_annotator", "components")


def component(name):
    def component_decorator(func):
        @functools.wraps(func)
        def component_factory(*args, **kwargs):
            return functools.partial(func, *args, **kwargs)

        # Make the factory signature identical to func's except for the first argument.
        signature = inspect.signature(func)
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        component_factory.__signature__ = signature
        components.register(name)(component_factory)
        return func

    return component_decorator


def valid_kwargs(kwargs: dict[str, Any], func: Callable) -> dict[str, Any]:
    args = list(inspect.signature(func).parameters)
    return {k: kwargs[k] for k in kwargs if k in args}


def find_latest_dir(directory_path: str):
    try:
        all_dirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' was not found.")
        return None

    latest_dir = None
    latest_date = None

    # Parse each directory name and find the latest
    for dir_name in all_dirs:
        try:
            # Try to parse the directory name as a date using the expected format
            dir_date = datetime.datetime.strptime(dir_name, "%Y-%m-%d_%H:%M:%S")

            # Update latest if this is the first valid directory or if it's newer
            if latest_date is None or dir_date > latest_date:
                latest_date = dir_date
                latest_dir = dir_name
        except ValueError:
            # Skip directories that don't match the expected format
            continue

    if latest_dir is None:
        print("No directories matching the format '%Y-%m-%d_%H:%M:%S' were found.")
        return None

    return latest_dir


def load_batch_files(batch_dir: str = None, feature: str = "") -> Dict:
    try:
        batch_dir = f"result/{feature}/batch_meta" if not batch_dir else batch_dir
        latest_dir = os.path.join(batch_dir, find_latest_dir(batch_dir))

        batch_list = [d for d in os.listdir(latest_dir)]
        batches = {}

        for batch_file in batch_list:
            # Extract the model name (everything before .json)
            model = batch_file.replace('.json', '')

            # Read and parse the content from the file
            try:
                if model == "gpt-4o":
                    with open(os.path.join(latest_dir, batch_file), 'r') as f:
                        data = json.load(f)
                        batches[model] = Batch.from_dict(data)
                elif model == "claude-3-7":
                    with open(os.path.join(latest_dir, batch_file), 'r') as f:
                        data = json.load(f)
                        batches[model] = Batch.from_dict(data)
            except Exception as e:
                print(f"Error reading file {batch_file}: {str(e)}")

        return batches
    except:
        raise FileNotFoundError("No batch file is found.")


def load_meta_file(batch_dir: str, feature: str):
    if not batch_dir and not feature:
        raise ValueError("Input batch_dir and feature cannot both be empty.")

    batch_dir = f"result/{feature}/batch_meta" if not batch_dir else batch_dir
    latest_dir = os.path.join(batch_dir, find_latest_dir(batch_dir))
    metadata_path = os.path.join(latest_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except:
            raise FileNotFoundError("Cannot find metadata.json.")
    return metadata


def create_batch_dir(feature: str, timestamp: str):
    results_dir = "result"
    feature_dir = os.path.join(results_dir, feature)
    os.makedirs(feature_dir, exist_ok=True)

    # Create batch dir
    batch_dir = os.path.join(feature_dir, f"{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    return batch_dir