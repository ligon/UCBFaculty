def identify_case(word):
    """
    Identify the case style of the input word.

    Args:
        word (str): The word whose case style is being identified.

    Returns:
        str: A string indicating the case style, such as 'uppercase', 'lowercase',
             'titlecase', or 'camelcase'.
    """
    if word.isupper():
        return 'uppercase'
    elif word.islower():
        return 'lowercase'
    elif word.istitle():
        return 'titlecase'
    elif word and word[0].islower() and not word.islower():
        return 'camelcase'
    else:
        return 'unknown'

## Usage example
#print(identify_case("HELLO"))     # Outputs: 'uppercase'
#print(identify_case("hello"))     # Outputs: 'lowercase'
#print(identify_case("Hello"))     # Outputs: 'titlecase'
#print(identify_case("helloWorld")) # Outputs: 'camelcase'

import json
from collections import defaultdict, UserDict
from ucbfaculty import spelling_dictionary

class StrippedDefaultDict(defaultdict):
    """A defaultdict that returns a stripped string of the missing key.

    Optionally holds a back-reference to a Dictionary so that accessing
    entries triggers lazy loading of the spelling data.
    """
    _owner = None  # set by Dictionary to enable lazy loading

    def _ensure_owner_loaded(self):
        if self._owner is not None:
            self._owner._ensure_loaded()

    def __missing__(self, key):
        return str(key).strip()

    def __getitem__(self, key):
        self._ensure_owner_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_owner_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_owner_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_owner_loaded()
        return super().__len__()

    def keys(self):
        self._ensure_owner_loaded()
        return super().keys()

    def items(self):
        self._ensure_owner_loaded()
        return super().items()

    def values(self):
        self._ensure_owner_loaded()
        return super().values()

class Dictionary(defaultdict):
    """A dictionary that manages canonical spellings and their variations."""

    def __init__(self, spelling_file=spelling_dictionary):
        """
        Initialize the dictionary using data from a JSON file.

        Loading is deferred until first access so that importing the module
        is cheap and won't crash if the file is missing.

        Args:
            spelling_file (str): Path to the JSON file containing the spelling dictionary.
        """
        super().__init__(list)
        self.spelling_file = spelling_file
        self.inverse_d = StrippedDefaultDict()
        self.inverse_d._owner = self
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        with open(self.spelling_file, 'r') as f:
            self.update(json.load(f))
        # Populate inverse_d in-place to preserve references captured at import time
        self.inverse_d.clear()
        for canonical, spellings in self.items():
            for spelling in spellings:
                self.inverse_d[spelling] = canonical
                self.inverse_d[spelling.title()] = canonical

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def add(self, key, value):
        """Register *value* as an alternate spelling of *key*.

        Parameters
        ----------
        key : str
            The canonical spelling.
        value : str
            A variant spelling to associate with *key*.
        """
        self._ensure_loaded()
        if value not in self[key]:
            if key not in self.keys():
                self[key].append(key)
            self[key].append(value)
            self.inverse_d[value] = key

    def persist(self, spelling_file=None):
        """Write the current dictionary to disk as JSON.

        Parameters
        ----------
        spelling_file : str | None
            Path to write.  Defaults to the file used at construction.
        """
        self._ensure_loaded()
        if spelling_file is None:
            spelling_file = self.spelling_file
        with open(spelling_file,'w') as f:
            json.dump(self,f,indent=4,sort_keys=True)

    def spell(self, word, preserve_case=False):
        """Return the canonical version of *word*.

        Parameters
        ----------
        word : str
            The word to normalise.
        preserve_case : bool
            When ``True``, the canonical form is adjusted to match
            the case style of the input (uppercase, lowercase, or
            title case).

        Returns
        -------
        str
            The canonical spelling.  Falls back to the stripped
            (and optionally title-cased) input when no mapping exists.
        """
        self._ensure_loaded()
        if preserve_case:
            word_case = identify_case(word)

        # NB: ``inverse_d`` is a ``StrippedDefaultDict`` whose ``__missing__``
        # silently returns the stripped key instead of raising ``KeyError``,
        # so we cannot rely on ``try/except KeyError`` here.  Use explicit
        # membership tests so the title-case fallback actually fires.
        if word in self.inverse_d:
            canonical = self.inverse_d[word]
        elif word.strip().title() in self.inverse_d:
            canonical = self.inverse_d[word.strip().title()]
        else:
            if preserve_case:
                return word.strip()
            else:
                return word.strip().title()

        if preserve_case:
            if word_case=='uppercase':
                canonical = canonical.upper()
            elif word_case=='lowercase':
                canonical = canonical.lower()
            elif word_case=='titlecase':
                canonical = canonical.title()

        return canonical


    def spellings(self, word):
        """
        Return all spellings for the given word.

        Args:
            word (str): The word to look up.

        Returns:
            list: All spellings for the word version, or an empty list if none are found.
        """
        self._ensure_loaded()
        try:
            return self[self.spell(word)]
        except KeyError:
            return [word]

    def spellings_map(self, word):
        """Return a dict mapping every known spelling of *word* to *word*.

        Parameters
        ----------
        word : str
            The canonical word to look up.

        Returns
        -------
        dict[str, str]
            ``{variant: word}`` for each variant, or ``{word: word}``
            when no spellings are found.
        """
        self._ensure_loaded()
        try:
            return {s:word for s in self.spellings(word)}
        except KeyError:
            return {word:word}

dictionary = Dictionary()
spell = dictionary.spell
spellings = dictionary.spellings
inverse_d = dictionary.inverse_d
spellings_map = dictionary.spellings_map
