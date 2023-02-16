export function sendRegistrationSuccessToIncomeAccess( hashedCustId: string ) {
  try {
    const frameElement = document.createElement("iframe")
    frameElement.src = `https://wlsbotop.adsrv.eacdn.com/Processing/Pixels/Registration.ashx?PlayerID=${ hashedCustId }&mid=2`
    frameElement.style.display = "none"
    document.body.appendChild(frameElement)
  }
  catch (e) {
    console.log("Load IA ad pixel failed!", e)
  }
}
