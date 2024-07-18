from ..dssdataloader import DSSDataLoader
from abc import abstractmethod
import logging

class NonAdaptiveDSSDataLoader(DSSDataLoader):
    """
    Implementation of NonAdaptiveDSSDataLoader class which serves as base class for dataloaders of other
    nonadaptive subset selection strategies for supervised learning setting.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    val_loader: torch.utils.data.DataLoader class
        Dataloader of the validation dataset
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, val_loader, dss_args, logger, *args,
                 **kwargs):
        """
        Constructor function
        """
        # Arguments assertion
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initialized = False
        self.device = dss_args.device
        super(NonAdaptiveDSSDataLoader, self).__init__(train_loader.dataset, dss_args,
                                                       logger, *args, **kwargs)

    def __iter__(self):
        """
        Iter function that returns the iterator of the data subset loader.
        """
        self.resample()
        return self.subset_loader.__iter__()
    
    def resample(self):
        """
        Function that resamples the subset indices and recalculates the subset weights
        """
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        print(len(self.subset_indices))
        self.logger.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader()
        self.logger.debug("Subset loader initiated, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        self.logger.debug('Subset selection finished, Training data size: %d, Subset size: %d',
                     self.len_full, len(self.subset_loader.dataset))

    @abstractmethod
    def _resample_subset_indices(self):
        """
        Abstract function that needs to be implemented in the child classes. 
        Needs implementation of subset selection implemented in child classes.
        """
        raise Exception('Not implemented.')
