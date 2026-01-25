"""
Download official biomedical ontologies (OBO format)

This script ONLY downloads ontology files, does not generate embeddings.
Use generate_ontology_embeddings.py for embedding generation.

Official sources:
- HPO: https://hpo.jax.org/app/data/ontology
- GO: http://geneontology.org/docs/download-ontology/
- MONDO: https://mondo.monarchinitiative.org/
- UBERON: http://obophenotype.github.io/uberon/
"""
import os
import requests
from tqdm import tqdm
from typing import Optional


class OntologyDownloader:
    """Download official ontology files"""
    
    ONTOLOGY_URLS = {
        'hpo': 'https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo',
        'go': 'http://purl.obolibrary.org/obo/go/go-basic.obo',
        'mondo': 'http://purl.obolibrary.org/obo/mondo.obo',
        'uberon': 'http://purl.obolibrary.org/obo/uberon/basic.obo'
    }
    
    @classmethod
    def download_ontology(cls, ontology_name: str, output_dir: str) -> Optional[str]:
        """
        Download an ontology file
        
        Args:
            ontology_name: Name of ontology (hpo, go, mondo, uberon)
            output_dir: Directory to save the file
            
        Returns:
            Path to downloaded file or None if failed
        """
        if ontology_name not in cls.ONTOLOGY_URLS:
            print(f"Unknown ontology: {ontology_name}")
            return None
        
        url = cls.ONTOLOGY_URLS[ontology_name]
        output_file = os.path.join(output_dir, f"{ontology_name}.obo")
        
        print(f"Downloading {ontology_name.upper()} from {url}...")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=ontology_name.upper()) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Downloaded {ontology_name.upper()} to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"✗ Failed to download {ontology_name}: {e}")
            return None
    
    @classmethod
    def download_all(cls, output_dir: str) -> dict:
        """
        Download all available ontologies
        
        Args:
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping ontology name to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded = {}
        for onto in cls.ONTOLOGY_URLS.keys():
            obo_file = os.path.join(output_dir, f"{onto}.obo")
            if os.path.exists(obo_file):
                print(f"✓ {onto.upper()} already exists: {obo_file}")
                downloaded[onto] = obo_file
            else:
                result = cls.download_ontology(onto, output_dir)
                if result:
                    downloaded[onto] = result
        
        return downloaded


def main():
    """Main function to download all ontologies"""
    
    # Configuration - download to InputsAndOutputs/input/ontologies
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(repo_root, "models", "InputsAndOutputs", "input", "ontologies")
    
    print("="*60)
    print("DOWNLOADING OFFICIAL BIOMEDICAL ONTOLOGIES")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print()
    
    # Download all ontologies
    downloaded = OntologyDownloader.download_all(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded: {len(downloaded)}/{len(OntologyDownloader.ONTOLOGY_URLS)}")
    for onto, path in downloaded.items():
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"  - {onto.upper()}: {path} ({file_size:.1f} MB)")
    
    print("\n✓ Download complete!")
    print(f"\nNext step: Run generate_ontology_embeddings.py to generate embeddings")


if __name__ == "__main__":
    main()
