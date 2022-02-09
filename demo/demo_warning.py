import warnings
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

# warnings.simplefilter('ignore')

def get_user_ip(url):
    if "http://" in url:
        warnings.warn("Deprecation warning. Using insecure api endpoint. Please use https url. Support for insecure requests will be removed at the end of 2016.", FutureWarning, stacklevel=2)
    request = urllib2.urlopen(url)
    response = request.read()
    print (response)
    return response

#calling in some other module 
def run():
    get_user_ip("http://httpbin.org/ip")

run()