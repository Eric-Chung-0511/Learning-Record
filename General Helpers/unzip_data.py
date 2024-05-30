# download the file you want to use
!wget http://example.com/somefile.zip

# unzip the file
import zipfile

def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Parameters:
    filename (str): A filepath to a target zip folder to be unzipped.

    Returns:
    None
    """
    # Open the zip file in read mode
    zip_ref = zipfile.ZipFile(filename, "r")
    
    # Extract all contents of the zip file into the current working directory
    zip_ref.extractall()
    
    # Close the zip file to free up resources
    zip_ref.close()

    # Or use with syntax, will automatically close the file
    #  with zipfile.ZipFile(filename, "r") as zip_ref:
        # zip_ref.extractall()


# Example Usage
# unzip_data('somefile.zip')
