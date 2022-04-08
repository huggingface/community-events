from pathlib import Path
from re import TEMPLATE
from typing import Optional, Union
import os

from huggingface_hub import PyTorchModelHubMixin, HfApi, HfFolder, Repository

from huggan import TEMPLATE_MODEL_CARD_PATH


class HugGANModelHubMixin(PyTorchModelHubMixin):
    """A mixin to push PyTorch Models to the Hugging Face Hub. This
    mixin was adapted from the PyTorchModelHubMixin to also push a template
    README.md for the HugGAN sprint.
    """

    def push_to_hub(
        self,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        commit_message: Optional[str] = "Add model",
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        api_endpoint: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
        config: Optional[dict] = None,
        skip_lfs_files: bool = False,
        default_model_card: Optional[str] = TEMPLATE_MODEL_CARD_PATH
    ) -> str:
        """
        Upload model checkpoint or tokenizer files to the Hub while
        synchronizing a local clone of the repo in `repo_path_or_name`.
        Parameters:
            repo_path_or_name (`str`, *optional*):
                Can either be a repository name for your model or tokenizer in
                the Hub or a path to a local folder (in which case the
                repository will have the name of that local folder). If not
                specified, will default to the name given by `repo_url` and a
                local directory with that name will be created.
            repo_url (`str`, *optional*):
                Specify this in case you want to push to an existing repository
                in the hub. If unspecified, a new repository will be created in
                your namespace (unless you specify an `organization`) with
                `repo_name`.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"add config"`,
                `"add tokenizer"` or `"add model"` depending on the type of the
                class.
            organization (`str`, *optional*):
                Organization in which you want to push your model or tokenizer
                (you must be a member of this organization).
            private (`bool`, *optional*):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                If `True`, will use the token generated when running
                `transformers-cli login` (stored in `~/.huggingface`). Will
                default to `True` if `repo_url` is not specified.
            git_user (`str`, *optional*):
                will override the `git config user.name` for committing and
                pushing files to the hub.
            git_email (`str`, *optional*):
                will override the `git config user.email` for committing and
                pushing files to the hub.
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            default_model_card (`str`, *optional*):
                Path to a markdown file to use as your default model card.
        Returns:
            The url of the commit of your model in the given repository.
        """

        if repo_path_or_name is None and repo_url is None:
            raise ValueError(
                "You need to specify a `repo_path_or_name` or a `repo_url`."
            )

        if use_auth_token is None and repo_url is None:
            token = HfFolder.get_token()
            if token is None:
                raise ValueError(
                    "You must login to the Hugging Face hub on this computer by typing `huggingface-cli login` and "
                    "entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own "
                    "token as the `use_auth_token` argument."
                )
        elif isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = None

        if repo_path_or_name is None:
            repo_path_or_name = repo_url.split("/")[-1]

        # If no URL is passed and there's no path to a directory containing files, create a repo
        if repo_url is None and not os.path.exists(repo_path_or_name):
            repo_id = Path(repo_path_or_name).name
            if organization:
                repo_id = f"{organization}/{repo_id}"
            repo_url = HfApi(endpoint=api_endpoint).create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                repo_type=None,
                exist_ok=True,
            )

        repo = Repository(
            repo_path_or_name,
            clone_from=repo_url,
            use_auth_token=use_auth_token,
            git_user=git_user,
            git_email=git_email,
            skip_lfs_files=skip_lfs_files
        )
        repo.git_pull(rebase=True)

        # Save the files in the cloned repo
        self.save_pretrained(repo_path_or_name, config=config)

        model_card_path = Path(repo_path_or_name) / 'README.md'
        if not model_card_path.exists():
            model_card_path.write_text(TEMPLATE_MODEL_CARD_PATH.read_text())

        # Commit and push!
        repo.git_add()
        repo.git_commit(commit_message)
        return repo.git_push()
